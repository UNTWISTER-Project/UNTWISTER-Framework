from flask import Flask, request, jsonify
import pika
import threading
import requests
import json
from collections import defaultdict
from threading import Lock
import csv
import os
import sys

app = Flask(__name__)

# Global variable to track configuration success
config_successful = False
queues_to_listen = []  # List of queues to listen to

latest_cpes_data = {}  # Solo dati da CPES (es. ego car, lead car reali)

CSV_RESULT_DIRECTORY = "/home/nico/Desktop/test/FINALGRAPHS/plot"
csv_header_written = False

# Detection and Mitigation component URL
detection_classification_url = "http://localhost:5002/start_work"

# Create a lock to protect access to the shared data buffer
data_buffer_lock = threading.Lock()
latest_cpes_data_lock = Lock()

# Buffer to store data by simstep, dynamically created based on sensor configurations
data_buffer = defaultdict(dict)  # Initialize an empty dict for each simstep

# Global variable to track the latest simstep
latest_forwarded_simstep = -1.0
forward_lock = Lock()  # To synchronize access to latest_simstep

def consume_messages(exchange_name, routing_key_name, queue_name):
    global csv_output_directory
    """Function to consume messages from a specified RabbitMQ queue."""
    connection_params = pika.ConnectionParameters('localhost', credentials=pika.PlainCredentials('guest', 'guest'))
    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()

    print("Declaring exchange")
    channel.exchange_declare(exchange=exchange_name, exchange_type='direct')

    # Declare the queue (ensure it exists)
    channel.queue_declare(queue=queue_name, exclusive=True)

    #print(f"I {exchange_name} bind to the routing key {routing_key_name}")

    channel.queue_bind(exchange=exchange_name, queue=queue_name,
                    routing_key=routing_key_name)

    # Callback function to process messages
    def callback(ch, method, properties, body):
        global latest_forwarded_simstep
        if "waiting for input data for simulation" in str(body):
            return
        
        data = json.loads(body)

        #print(f"Received message from {queue_name}: {data}")
        
        # Get simstep
        simstep = data.get('simstep', None)

        if simstep is None:
            print("Error: Message missing 'simstep'")
            return
        
        
        if routing_key_name == "sim1.data.from_DT":
            if csv_output_directory != None:
                write_data_to_csv(simstep, data, True)
        else:
            if csv_output_directory != None: 
                write_data_to_csv(simstep, data, False)
            with latest_cpes_data_lock:
                latest_cpes_data["simstep"] = simstep
                latest_cpes_data["data"] = {key: value for key, value in data.items() if key != "simstep" and key != "time"}

        float_simstep = float(simstep.replace(',', '.'))
        
        # Check for missing or outdated simstep
        with forward_lock:  # Protect access to latest_forwarded_simstep
            if float_simstep <= latest_forwarded_simstep:
                print(f"Error: outdated (simstep: {simstep}, latest: {latest_forwarded_simstep})")
                return
        
        # Extract all key-value pairs except 'simstep'
        sensor_data = {key: value for key, value in data.items() if key != 'simstep' and key != 'time'}

        # Synchronize access to the shared data_buffer using a lock
        with data_buffer_lock:
            # Store the sensor data in the buffer dynamically
            data_buffer[simstep][queue_name] = sensor_data
            # Check if we have received all data for this simstep
            all_data_available = all(queue in data_buffer[simstep] for queue in queues_to_listen)
        if all_data_available:
            #print(f"Data for simstep {simstep}: {data_buffer[simstep]}")

            # Combine all the sensor data for this simstep across all queues
            combined_data = {}
            for queue, sensor in data_buffer[simstep].items():
                combined_data.update(sensor)  # Merge each queue's data into combined_data

            # Forward data to detection and mitigation component
            forward_to_detection_classification(combined_data, simstep)

            # Update latest_forwarded_simstep after successful forwarding
            with forward_lock:
                latest_forwarded_simstep = float_simstep
                #print(f"Updated latest forwarded simstep to {latest_forwarded_simstep}")

    # Start consuming messages
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    print(f"Listening for messages on queue: {queue_name}")
    channel.start_consuming()

def forward_to_detection_classification(data, simstep):
    """Function to forward data to the detection and mitigation component."""
    json_data = {
        "simstep": simstep,
        "data": data
    } 
    try:
        # Now send the data to the detection component
        response = requests.post(detection_classification_url, json=json_data)

        """ if response.status_code == 200:
            print(f"Data forwarded to detection component: {json_data}")
        else:
            print(f"Failed to forward data, status code: {response.status_code}") """
    except requests.RequestException as e:
        print(f"Error forwarding data to detection component: {e}")


def write_data_to_csv(simstep: str, data: dict, isDT: bool):
    """Append combined data to a CSV file with one row per simstep."""
    global csv_header_written, csv_output_directory

    data_with_simstep = {"simstep": simstep}
    data_with_simstep.update(data)  # Add simstep as first column

    filename = "DT.csv" if isDT else "PS.csv"
    full_csv_path = os.path.join(csv_output_directory, filename)

    write_header = not os.path.exists(full_csv_path) or not csv_header_written

    with open(full_csv_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_with_simstep.keys())

        if write_header:
            writer.writeheader()
            csv_header_written = True

        writer.writerow(data_with_simstep)

@app.route('/latest_cpes_data', methods=['GET'])
def get_latest_cpes_data():
    with latest_cpes_data_lock:
        if not latest_cpes_data:
            return jsonify({"status": "error", "message": "No CPES data available yet"}), 404
        return jsonify({
            "status": "success",
            "latest_cpes_data": latest_cpes_data
        })

@app.route('/data_collection', methods=['POST'])
def data_collection():
    global config_successful, sensor_configurations, queues_to_listen
    
    config_data = request.json
    print(f"Data Collection Component received configuration: {config_data}")
    
    # Check if the configuration is valid
    if "sensors" in config_data:

        config_successful = True
        # Extract queue names along with their respective exchanges and routing keys
        sensor_configurations = [
            {
                "exchange": sensor["exchange"],
                "queue": sensor["queue"],
                "routing_key": sensor["routing_key"]
            }
            for sensor in config_data["sensors"].values()
        ]

        # estrai i nomi delle code da ascoltare
        queues_to_listen = [sensor["queue"] for sensor in sensor_configurations]

        return jsonify({"status": "success", "message": "Data Collection configured successfully."})
    else:
        config_successful = False
        return jsonify({"status": "failure", "message": "Configuration failed due to missing parameters."}), 400

@app.route('/start_work', methods=['POST'])
def start_work():
    global config_successful, sensor_configurations
    
    if config_successful:
        print("Data Collection Component starting working phase...")

        # Start listening to the queues in separate threads
        for sensor in sensor_configurations:
            exchange = sensor["exchange"]
            queue_name = sensor["queue"]
            # For each sensor, initialize a dictionary for each simstep
            data_buffer[queue_name] = {}
            routing_key = sensor["routing_key"]

            # Start consuming messages for each sensor
            threading.Thread(target=consume_messages, args=(exchange, routing_key, queue_name), daemon=True).start()

        return jsonify({"status": "success", "message": "Working phase started."})
    else:
        return jsonify({"status": "failure", "message": "Component not properly configured. Cannot start."}), 400

if __name__ == "__main__":
    csv_output_directory = None
    if len(sys.argv) >= 2:
        try:
            seed = int(sys.argv[1])
            print(f"Filename specified: {seed}")
            # Crea directory specifica per il seed
            seed_dir = os.path.join(CSV_RESULT_DIRECTORY, str(seed))
            os.makedirs(seed_dir, exist_ok=True)
            csv_output_directory = seed_dir
        except ValueError:
            print("Not valid argument, it will be ignored.")
    else:
        print("Filename not specified, proceding without filename.")
    app.run(port=5001)  # Run on port 5001
