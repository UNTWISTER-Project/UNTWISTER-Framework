from flask import Flask, request, jsonify
import torch
import numpy as np
import os
import requests
import pandas as pd
from model_creation.models import MLP
import csv
import sys
import pika
import json 
from datetime import datetime, timezone

app = Flask(__name__)

# Global variables to store the model
MLPmodel = None

ema_reg_value = None  # inizializza EMA come None
alpha = 0.3  # fattore di smoothing (puoi regolarlo)
true_attack = False
potential_attack = False

# === FINESTRA MOBILE E INFERENZA ===
window_size = 500
step_size = 5
df = pd.DataFrame()

counter = 0
false_positive = 0
stop_processing = 0

# Parameters for MLP model
mlp_input_size = 10
num_classes = 3   

# Path to temporarily store the uploaded ZIP file
UPLOAD_FOLDER = "detection_and_classification/uploads"
ATTACK_MITIGATION = "http://localhost:5003"
PREDICTION_LOG_PATH = "/home/nico/Desktop/test/FINALGRAPHS/plot/predictions.csv"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable to track configuration success
config_successful = False

def send_attack_classification(mitigation_url, attack_class, attack_parameter, state_variables):
    """
    Send the attack classification to the attack mitigation component.
    """
    try:
        payload = {
            "attack_class": int(attack_class),
            "attack_parameter": attack_parameter,
            "state_variables": state_variables
        }
        print(f"Sending attack classification to {mitigation_url}: {payload}")
        response = requests.post(mitigation_url, json=payload)
        return response
    except Exception as e:
        print(f"Error sending attack classification: {e}")
        return None
    
def log_prediction(seed, simstep, predicted_class, ema_reg_value, output_path=PREDICTION_LOG_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_header = not os.path.exists(output_path)

    with open(output_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "simstep", "predicted_class", "ema_reg_value"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "seed": seed,
            "simstep": simstep,
            "predicted_class": predicted_class,
            "ema_reg_value": ema_reg_value
        })

def send_signal_to_PS(signal):
    # Create connection to rabbitMQ server
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    try:
        timestamp = datetime.now(timezone.utc).astimezone().isoformat()
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
        print("STOP MESSAGE INVIATO")
    except Exception as e:
        print(" [!] Errore nel parsing del messaggio:", e)

@app.route('/start_work', methods=['POST'])
def start_work():
    global config_successful, window_size, step_size, df, MLPmodel, stop_processing, \
    counter, false_positive, true_attack, potential_attack, ema_reg_value, alpha, seed

    if not config_successful:
        print("Component not properly configured. Cannot start working phase.")
        return jsonify({"status": "failure", "message": "Component not properly configured."}), 400

    # Step 1: Receive and preprocess the input data
    received_data = request.json
    #print(f"Received data: {received_data}")

    if stop_processing:
        return jsonify({"status": "success", "message": "Anomaly already detected."}), 200 

    # Parse simstep from the incoming data
    simstep = float(received_data['simstep'].replace(',', '.'))
    if simstep is None:
        return jsonify({"status": "failure", "message": "Missing 'simstep' in the data."}), 400  

    # Extract the 'data' part from the received data, which contains the sensor data
    sensor_data = received_data.get('data', {})

   # Convert the dictionary (sensor_data) into a DataFrame (single row DataFrame)
    sensor_data_df = pd.DataFrame([sensor_data])

    # Check if the DataFrame is empty, i.e., first time data is being received
    if df.empty:
        # If df is empty, create a new DataFrame using the keys of the first incoming data as column names
        df = pd.DataFrame(columns=sensor_data.keys())

    # Check for NaN values in the sensor_data_df (after converting to DataFrame)
    if sensor_data_df.isna().any().any():  # This checks if there are any NaN values in the DataFrame
        return jsonify({"status": "failure", "message": "Sensor data contains NaN values."}), 400

    # Append the new DataFrame to the global DataFrame
    df = pd.concat([df, sensor_data_df], ignore_index=True)
    # Increment the counter
    counter += 1

    #print(f"dim df: {len(df)}, counter: {counter}")
    # Se abbiamo abbastanza dati per una finestra
    if len(df) >= window_size and counter % step_size == 0:

        # Estrai l'ultima finestra
        window_df = df.iloc[-window_size:].copy()
        #print(f"dimensione finestra dopo la copia: {len(window_df)}")

        # Calcoli differenze
        window_df["ego_x_difference"] = window_df["ego_x"] - window_df["DT_ego_x"]
        window_df["ego_speed_difference"] = window_df["ego_speed"] - window_df["DT_ego_speed"]
        window_df["cacc_acc_difference"] = window_df["cacc_accel"] - window_df["DT_cacc_accel"]
        window_df["distance"] = window_df["lead_x"] - window_df["ego_x"]
        window_df["speed_difference"] = window_df["lead_speed"] - window_df["ego_speed"]

        # Estrazione delle feature
        features = []
        feature_cols = [
            "ego_x_difference",
            "ego_speed_difference",
            "cacc_acc_difference",
            "distance",
            "speed_difference"
        ]

        for col in feature_cols:
            mean_val = window_df[col].mean()
            var_val = window_df[col].var()
            features.append(mean_val)
            features.append(var_val)

        # Converti in numpy array 1D
        features_np = np.array(features).reshape(1, -1)

        # Crea un DataFrame con i nomi delle colonne
        feature_names = []
        for col in feature_cols:
            feature_names.append(f"{col}_mean")
            feature_names.append(f"{col}_var")
        features_df = pd.DataFrame(features_np, columns=feature_names)

        ordered_columns = [
        "ego_x_difference_mean", "ego_speed_difference_mean", "cacc_acc_difference_mean",
        "distance_mean", "speed_difference_mean",
        "ego_x_difference_var", "ego_speed_difference_var", "cacc_acc_difference_var",
        "distance_var", "speed_difference_var"
        ]

        # Riordina il DataFrame secondo la lista sopra
        features_df = features_df[ordered_columns]
        features_df.dropna(inplace=True)

        # Converti in tensore PyTorch
        input_tensor = torch.tensor(features_df.values, dtype=torch.float32)

        # Esegui inferenza
        with torch.no_grad():
            class_output, reg_output = MLPmodel(input_tensor)
            predicted_class = torch.argmax(class_output, dim=1).item()
            #print(class_output, predicted_class)
            reg_value = reg_output.item()

            # print("Predizione:", predicted_class, "| Regressione:", reg_value)

            if predicted_class != 0:
                print("Anomalia rilevata!")
                potential_attack = True
                false_positive += 1
                if ema_reg_value is None:
                    ema_reg_value = reg_value  # primo valore, inizializza
                else:
                    ema_reg_value = alpha * reg_value + (1 - alpha) * ema_reg_value  # aggiorna
                if false_positive == 7:
                    true_attack = True
            else:
                potential_attack = False
            if (true_attack and false_positive == 10) or (true_attack and potential_attack == False):
                stop_processing = True 

                send_signal_to_PS(True)

                # Step 5: calculate the state variables
                state_variables = {}

                state_variables_config  = config_data["state_variables"]
                # Process raw variables
                for var in state_variables_config["raw"]:
                    state_variables[var] = received_data['data'].get(var, None)

                # Process composed variables
                for composed_var, expression in state_variables_config["composed"].items():            
                    composed_value = eval(expression)  # Evaluating the expression
                    state_variables[composed_var] = composed_value

                # Log the prediction if the seed is specified
                if seed is not None:
                    log_prediction(seed, composed_value, int(predicted_class), ema_reg_value)

                # Step t: Send attack classification to `attack_mitigation` component
                mitigation_url = f"{ATTACK_MITIGATION}/mitigate"
                response = send_attack_classification(mitigation_url, int(predicted_class),
                                                        ema_reg_value, state_variables) 

                if response.status_code == 200:
                    return jsonify({
                        "status": "success",
                        "message": "Attack classification sent successfully.",
                        "classification": int(predicted_class),
                        "attack_parameter": ema_reg_value, 
                        "state_variables": state_variables
                    }), 200
                else:
                    print(f"Error sending attack classification: {response.text}")
                    return jsonify({"status": "failure", "message": "Failed to send attack classification."}), 500

        # clean up
        if len(df) > window_size * 2:
            df = df.iloc[-window_size * 2:].reset_index(drop=True)

        return jsonify({"status": "success", "message": "Message received."}), 200    
    else:
        return jsonify({"status": "success", "message": "Message received."}), 200    

@app.route('/detection_classification', methods=['POST'])
def detection_classification_config():
    global config_data, config_successful

    # Receive and validate JSON configuration
    config_data = request.json
    required_keys = ["mlp_features", "state_variables"]
    
    for key in required_keys:
        if key not in config_data:
            config_successful = False
            return jsonify({"status": "failure", "message": f"Missing required key: {key}"}), 400
    
    # Save configuration and set flag
    config_successful = True
    return jsonify({"status": "success", "message": "Configuration received successfully."}), 200


@app.route('/detection_classification/upload', methods=['POST'])
def detection_classification_upload():
    global config_successful, MLPmodel

    # Check if the configuration step has been completed
    if not config_successful:
        print("Configuration not completed. Returning failure response.")
        return jsonify({"status": "failure", "message": "Configuration not completed. Please configure first."}), 400

    # Check for the uploaded ZIP file
    if 'file' not in request.files or not request.files['file'].filename:
        print("No file uploaded. Returning failure response.")
        return jsonify({"status": "failure", "message": "No file uploaded."}), 400

    file = request.files['file']
    print(f"Received file: {file.filename}")

    # Check if it's the expected model file
    if file.filename != "mlp_modelWithParam.pth":
        return jsonify({"status": "failure", "message": "Expected file 'mlp_model.pth'."}), 400

    # Save the model temporarily
    model_path = os.path.join(UPLOAD_FOLDER, "mlp_model.pth")
    try:
        file.save(model_path)
        print(f"Model file saved to {model_path}")
    except Exception as e:
        print(f"Error saving model file: {e}")
        return jsonify({"status": "failure", "message": f"Failed to save model file: {e}"}), 500

    try:
        MLPmodel = MLP(mlp_input_size, num_classes)
        MLPmodel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        MLPmodel.eval()
        print("MLP model loaded successfully.")
    except Exception as e:
        print(f"Error loading MLP model: {e}")
        return jsonify({"status": "failure", "message": f"Error loading MLP model: {e}"}), 500

    # Indicate success
    print("Model uploaded and loaded successfully.")
    return jsonify({"status": "success", "message": "MLP model uploaded and loaded successfully."}), 200


if __name__ == "__main__":
    seed = None
    if len(sys.argv) >= 2:
        try:
            seed = int(sys.argv[1])
            print(f"Seed specificato: {seed}")
        except ValueError:
            print("Argomento seed non valido, sarà ignorato.")
    else:
        print("Nessun seed specificato, procedo senza seed.")
    app.run(port=5002, debug=False)  # Run on port 5002
