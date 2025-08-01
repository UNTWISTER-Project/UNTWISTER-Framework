from flask import Flask, request, jsonify
import pika
import datetime
import json
import pika
import numpy as np
import time
import requests

app = Flask(__name__)

# RabbitMQ connection parameters
RABBITMQ_HOST = 'localhost'
RABBITMQ_PORT = 5672
RABBITMQ_USERNAME = 'guest'
RABBITMQ_PASSWORD = 'guest'

data_and_subscription_url = "http://localhost:5001/latest_cpes_data"

# Global variable to track configuration success
config_successful = False

def send_message(queue_name, exchange, routing_key, message):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, virtual_host='/',
                                                                   credentials=pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD)))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    channel.basic_publish(exchange=exchange, routing_key=routing_key, body=json.dumps(message))
    print(f"Sent message to {queue_name}: {message}")
    connection.close()

def message_to_physical_system(command_lead, command_ego):
    # Get the current time in UTC
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    # Define the desired UTC offset (e.g., +02:00)
    offset = datetime.timedelta(hours=0)
    now_with_offset = now_utc.astimezone(datetime.timezone(offset))

    # Format the timestamp in ISO 8601 format
    timestamp_iso8601 = now_with_offset.isoformat()


    # Create a JSON object with the timestamp
    message = {
        "time": timestamp_iso8601,
        "stop_mode": False,
        "lead_acceleration": command_lead,
        "ego_acceleration": command_ego
    }
    print("the message created is: ", message)
    return message

def send_signal_to_PS(signal):
    # Create connection to rabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    try:
        timestamp = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
        new_msg = {
            'time': timestamp,
            'stop_mode': signal,
            'lead_acceleration': -1,
            'ego_acceleration': -1
        }
        #print(" [x] Pubblico:", new_msg)
        channel.basic_publish(
            exchange='fmi_digital_twin',
            routing_key='sim1.data.to_cosim',
            body=json.dumps(new_msg)
        )
    except Exception as e:
        print(" [!] Errore nel parsing del messaggio:", e)

@app.route('/start_work', methods=['POST'])
def start_work():
    global config_successful, config_data
    
    if config_successful:
        received_data = request.json
        if received_data is None:
            return jsonify({"status": "failure", "message": "empty message"}), 500
        
        print(received_data)
        
        # Extract command and sensor
        try:
            command = float(received_data.get("command"))
        except (TypeError, ValueError):
            return jsonify({"status": "failure", "message": "Invalid command value."}), 400
        
        sensor = received_data.get("sensor")
        if not sensor:
            return jsonify({"status": "failure", "message": "Sensor not specified."}), 400
        
        # Get sensor-specific configuration
        sensors_config = config_data.get("sensors", {})
        sensor_config = sensors_config.get(sensor)
        
        if not sensor_config:
            return jsonify({"status": "failure", "message": f"Configuration for sensor '{sensor}' not found."}), 400
        
        # Extract the parameters for the message
        exchange = sensor_config.get("exchange")
        queue = sensor_config.get("queue")
        routing_key = sensor_config.get("routing_key")
        
        if not all([exchange, queue, routing_key]):
            return jsonify({"status": "failure", "message": f"Incomplete configuration for sensor '{sensor}'."}), 400

        # time simulated for the prediction
        media = 1.5
        dev_std = 0.3

        valore_normale = np.random.normal(media, dev_std)

        print(f"tempo attesa {valore_normale}")

        send_signal_to_PS(False)

        # Simulazione artificiale del tempo
        time.sleep(valore_normale)
        print(f"Prendo l'ultimo dato {datetime.datetime.now()}")
        # Retrieve the last CPES data
        resp = requests.get(data_and_subscription_url)
        if resp.ok:
            latest_data = resp.json().get("latest_cpes_data", {})
            simstep = latest_data.get("simstep")
            print(f"Last simstep: {simstep}")
            cpes = latest_data.get("data", {})
            #print(f"Data: {cpes}")
            x_ego = float(cpes.get("ego_x", 0))
            v_ego = float(cpes.get("ego_speed", 0))
            x_lead = float(cpes.get("lead_x", 0))
            v_lead = float(cpes.get("lead_speed", 0))

            # Considerazione della distanza fisica tra veicoli (lunghezza auto)
            vehicle_length = 4.0  # metri (es. media auto compatta)
            body_clearance = vehicle_length  # 2 m davanti e 2 m dietro → totale 4 m

            # Distanza effettiva reale tra auto
            d0 = x_lead - x_ego
            d_eff = d0 - body_clearance  # distanza tra carrozzerie, non solo tra centri

            max_deceleration = 7.0  # limite fisico per frenata
            increment = 1.0  # passo con cui aumentare lo spazio

            # Distanza entro cui vogliamo che si fermino
            x_stop = 30.0  # metri

            # Itera finché entrambe le decelerazioni sono ammissibili
            while True:
                # Calcola le decelerazioni richieste
                a_ego = v_ego**2 / (2 * x_stop) if x_stop > 0 else float('inf')
                a_lead = v_lead**2 / (2 * x_stop) if x_stop > 0 else float('inf')

                if a_ego <= max_deceleration and a_lead <= max_deceleration:
                    print(f"Spazio trovato: {x_stop:.1f} m")
                    print(f"a_ego = {a_ego:.2f} m/s², a_lead = {a_lead:.2f} m/s²")
                    break  # condizione soddisfatta

                # Altrimenti, aumenta lo spazio
                print(f"❌ Decelerazione troppo alta. a_ego = {a_ego:.2f}, a_lead = {a_lead:.2f} → aumento spazio.")
                x_stop += increment
                
            try:
                safety_term = (v_ego**2 - v_lead**2) / (2 * a_ego)
            except ZeroDivisionError:
                safety_term = float('inf')

            # Verifica del criterio
            print(f"Distanza iniziale: {d0:.2f} m")
            print(f"Decelerazioni necessarie: ego = {a_ego:.2f} m/s², lead = {a_lead:.2f} m/s²")
            print(f"Soglia minima di sicurezza: {safety_term:.2f} m")

            if d_eff > safety_term:
                print("NESSUN rischio di scontro durante la frenata in 30 m.")
                # Create and send the message
                message = message_to_physical_system(command_lead=a_lead, command_ego=a_ego)

            else:
                print("RISCHIO di scontro: serve più distanza o frenata più forte!")
                while d_eff <= safety_term:
                    x_stop += increment
                    a_ego = v_ego**2 / (2 * x_stop)
                    a_lead = v_lead**2 / (2 * x_stop)
                    safety_term = (v_ego**2 - v_lead**2) / (2 * a_ego)
                    print(f"↪️ Nuovo x_stop: {x_stop:.1f} m, a_ego: {a_ego:.2f}, a_lead: {a_lead:.2f}, safety_term: {safety_term:.2f}")


            # Step 6: Log the prediction if the seed is specified
            """ if seed is not None:
                log_reconfig_command(seed, command_lead, command_ego, PREDICTION_LOG_PATH)
            else:
                if seed is not None:
                    log_reconfig_command(seed, -1.0, PREDICTION_LOG_PATH) """
        else:  
            return {"status": "error", "message": "Failed to retrieve the last simstep"}, 500 

        send_message(queue, exchange, routing_key, message)
        print(f"Message sent to {queue} via {exchange} with routing key {routing_key}: {message}")
        
        return jsonify({"status": "success", "message": "Message sent successfully."}), 200
    else:
        print("Component not properly configured. Cannot start working phase.")
        return jsonify({"status": "failure", "message": "Component not configured."}), 400


@app.route('/reconfiguration', methods=['POST'])
def reconfiguration():
    global config_data, config_successful
    
    config_data = request.json
    print(f"Reconfiguration Component received configuration: {config_data}")
    
    if "sensors" in config_data:
        config_successful = True
        return jsonify({"status": "success", "message": "Reconfiguration configured successfully."})
    else:
        config_successful = False
        return jsonify({"status": "failure", "message": "Configuration failed due to missing parameters."}), 400

if __name__ == "__main__":
    app.run(port=5005)  # Run on port 5005
