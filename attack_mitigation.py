from flask import Flask, request, jsonify
import json
import subprocess
import os
import http.client
import pandas as pd
import requests
import sys
import csv
import http.client
import json
import time

app = Flask(__name__)

CSV_RESULT = "./prediction.csv"
SIMULATE_JSON = "./simulate.json"
SAVE_DIRECTORY = "/home/nico/into-cps-projects/modelsPrediction/Multi-models/"
CSV_RESULT_DIRECTORY = "/home/nico/Desktop/test/FINALGRAPHS/plot"
PREDICTION_LOG_PATH = "/home/nico/Desktop/test/FINALGRAPHS/plot/predictions.csv"

reconfiguration_url = "http://localhost:5005/start_work"

# Ensure the directory exists
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

COE_PORT = 8084

# Global variable to track configuration success
config_successful = False

def modify_mm_json(state_variables, attack_parameter, attack_id, config_file):
    # Read the current mm.json
    with open(config_file, 'r') as f:
        mm_json = json.load(f)
    
    # Retrieve the mapping from 'json_modify'
    json_modify = config_data["json_modify"]

    # Update parameters dynamically based on state_variables and json_modify
    for json_path, variable_name in json_modify.items():
        if variable_name in state_variables:
            mm_json['parameters'][json_path] = state_variables[variable_name]
        else:
            print(f"Warning: Variable '{variable_name}' specified in json_modify not found in state_variables.")

    if attack_id == 2:
        mm_json['parameters']["{attack_plus_constant}.attack_plus_constantInstance.attackValue"] = attack_parameter
    else: 
        mm_json['parameters']["{ramp_attack}.ramp_attackInstance.rampValue"] = attack_parameter

    # Insert algorithm and simulationProgramDelay if not present
    if 'algorithm' not in mm_json:
        mm_json['algorithm'] = {"type": "fixed-step", "size": 0.01}
    
    if 'simulationProgramDelay' not in mm_json:
        mm_json['simulationProgramDelay'] = False

    # Write the updated mm.json back to the file
    with open(config_file, 'w') as f:
        json.dump(mm_json, f, indent=4)
    print("Modified mm.json with analysis_result")

def post_request(conn, location, data_path):
    headers = {'Content-type': 'application/json'}
    with open(data_path, 'r') as file:
        json_data = json.dumps(json.load(file))
    conn.request('POST', location, json_data, headers)
    return conn.getresponse()

def start_co_simulation(config_file_path, simulate_json_path, coe_port=8084, csv_output_path="results.csv", seed=1234):
    """Avvia una co-simulazione tramite le API del COE RESTful, misurando i tempi delle fasi principali."""
    conn = http.client.HTTPConnection(f'localhost:{coe_port}')

   # Se seed è specificato, salva in directory dedicata
    if seed is not None:
        seed_dir = os.path.join(CSV_RESULT_DIRECTORY, str(seed))
        os.makedirs(seed_dir, exist_ok=True)
        csv_output_path = os.path.join(seed_dir, "prediction.csv")
    else:
        # Altrimenti salva nella cartella corrente
        csv_output_path = os.path.join(os.getcwd(), "prediction.csv")
    
    try:
        # 1️ Crea sessione
        print("Creazione sessione...")
        conn.request('GET', '/createSession')
        response = conn.getresponse()
        if response.status != 200:
            raise Exception("Errore nella creazione della sessione")
        session_info = json.loads(response.read().decode())
        session_id = session_info["sessionId"]
        print(f"Sessione creata: {session_id}")

        # 2️ Inizializza simulazione
        print("Inizializzazione simulazione...")
        response = post_request(conn, f'/initialize/{session_id}', config_file_path)
        if response.status != 200:
            raise Exception(f"Inizializzazione fallita: {response.read().decode()}")
        print(f"Inizializzato: {response.read().decode()}")

        # 3️ Avvia simulazione
        print("Avvio simulazione...")
        response = post_request(conn, f'/simulate/{session_id}', simulate_json_path)
        if response.status != 200:
            raise Exception(f"Simulazione fallita: {response.read().decode()}")
        print(f"Simulazione completata: {response.read().decode()}")

        # 4️ Recupera risultati
        print("Recupero risultati...")
        conn.request('GET', f'/result/{session_id}/plain')
        response = conn.getresponse()
        if response.status != 200:
            raise Exception("Errore nel recupero dei risultati")
        csv_data = response.read().decode()

        with open(csv_output_path, "w") as f:
            f.write(csv_data)
        print(f"Risultati salvati in {csv_output_path}")

        # 5️ Distrugge sessione
        print("Pulizia sessione...")
        conn.request('GET', f'/destroy/{session_id}')
        response = conn.getresponse()
        if response.status != 200:
            raise Exception("Errore nella distruzione della sessione")
        print(f"Sessione chiusa: {response.read().decode()}")

    except Exception as e:
        print(f"Errore nella simulazione: {e}")
        return None
    finally:
        conn.close()

def analyze_forecasted_data_mode(critical_scenarios, seed):
    """
    Analyze forecasted data to determine if any critical conditions are met.

    Parameters:
        critical_scenarios (dict): Dictionary of critical scenarios with expressions.

    Returns:
        bool: True if any critical condition is met, False otherwise.
    """
    # Step 1: Read the CSV file into a pandas DataFrame
    if seed is not None:
        seed_dir = os.path.join(CSV_RESULT_DIRECTORY, str(seed))
        os.makedirs(seed_dir, exist_ok=True)
        csv_output_path = os.path.join(seed_dir, "prediction.csv")
        try:
            df = pd.read_csv(csv_output_path)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return False
    else:
        print(f"Reading CSV file: {CSV_RESULT}")
        try:
            df = pd.read_csv(CSV_RESULT)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return False

    # Step 2: Clean and rename columns to remove `{}` for easier referencing
    try:
        df_clean = df.dropna().copy()
        rename_mapping = {col: col.replace("{", "").replace("}", "") for col in df_clean.columns}
        df_clean.rename(columns=rename_mapping, inplace=True)
        
    except Exception as e:
        print(f"Error processing DataFrame columns: {e}")
        return False

    # Step 3: Iterate over critical scenarios and evaluate expressions
    for event_name, event_details in critical_scenarios.items():
        # Check if event_details is valid
        if event_details is None:
            print(f"Event details for {event_name} are None. Skipping.")
            continue
        
        # Extract the expression for the critical condition
        expression = event_details.get("expression", "")
        if not expression:
            print(f"No expression defined for {event_name}. Skipping.")
            continue
        try:
            # Replace placeholders in the expression with DataFrame column references
            for col in rename_mapping.keys():
                placeholder = col.replace("{", "").replace("}", "")
                expression = expression.replace(col, f'df_clean["{placeholder}"]')
            
            # Evaluate the expression
            critical_met = eval(expression)  # Check if the condition is True for any row
            
            if critical_met.any():
                # Extract time from the first row that meets the condition
                first_time = df_clean.loc[critical_met, "time"].iloc[0]
                print(f"Crash Time: {first_time}")
                return event_name, first_time, True

        except Exception as e:
            print(f"Error evaluating expression for {event_name}: {e}")

    print("No critical conditions met.")
    return None, False, None

def send_command_to_reconfiguration(command, sensor):
    """Function to forward data to the reconfiguration component."""
    json_data = {
        "command": command,
        "sensor": sensor
    } 
    try:
        response = requests.post(reconfiguration_url, json=json_data)
        return response
    except requests.RequestException as e:
        print(f"Error forwarding data to detection component: {e}")

@app.route('/mitigate', methods=['POST'])
def mitigate():
    global seed
    try:
        # Access the JSON payload
        payload = request.get_json()

        print(payload)        

        state_variables = payload['state_variables']
        attack_id = payload['attack_class']
        attack_parameter = payload['attack_parameter']
        # Construct the file path for the corresponding attack
        file_path = os.path.join(SAVE_DIRECTORY, f"attack_{attack_id}_conf_prediction.json")
        
        modify_mm_json(state_variables, attack_parameter, attack_id, file_path)

        start_co_simulation(file_path, SIMULATE_JSON, COE_PORT, CSV_RESULT, seed=seed)

        # Access the attack details
        attacks = config_data.get("attacks", {})

        attack = attacks.get(str(attack_id))
        if attack is None:
            print(f"No attack found for attack_id: {attack_id}.")
            return

        # Get the critical scenarios for the specified attack
        critical_scenarios = attack.get("critical_scenarios", {}) if attack else {}
        if not critical_scenarios:
            print(f"No critical scenarios found for attack_id: {attack_id}.")
            return

        event, crashTime, isCritical = analyze_forecasted_data_mode(critical_scenarios, seed)
        if isCritical:

            sensor = critical_scenarios[event]["sensor"]
            response = send_command_to_reconfiguration(1, sensor)

            if response.status_code == 200:
                print(f"Data forwarded to reconfiguration component")
                return {"status": "success", "message": "command sent to the reconfiguration component"}, 200

            else:
                print(f"Failed to forward data, status code: {response.status_code}")
                return {"status": "error", "message": "Failed to send the command"}, 500

        # Respond to the client
        return {"status": "success", "message": "Payload received and processed"}, 200
    except Exception as e:
        print(f"Error processing payload: {e}")
        return {"status": "error", "message": "Failed to process payload"}, 500

@app.route('/start_work', methods=['POST'])
def start_work():
    global config_successful
    if config_successful:
        print("Attack Mitigation Component starting working phase...")
    else:
        print("Component not properly configured. Cannot start working phase.")

@app.route('/attack_mitigation', methods=['POST'])
def attack_mitigation():
    global config_data, config_successful
    
    config_data = request.json
    print(f"Attack Mitigation Component received configuration: {config_data}")
        
    if  "attacks" in config_data:
         # Process and store conf_prediction for each attack
        attacks = config_data.get("attacks", {})
        for attack_id, attack_details in attacks.items():
            if "conf_prediction" in attack_details:
                conf_prediction = attack_details["conf_prediction"]
                # Write conf_prediction to a JSON file
                file_name = os.path.join(SAVE_DIRECTORY, f"attack_{attack_id}_conf_prediction.json")
                with open(file_name, 'w') as json_file:
                    json.dump(conf_prediction, json_file, indent=4)
                print(f"Stored conf_prediction for attack {attack_id} in {file_name}")
        config_successful = True
        return jsonify({"status": "success", "message": "Attack Mitigation configured successfully."})
    else:
        config_successful = False
        return jsonify({"status": "failure", "message": "Configuration failed due to missing parameters."}), 400

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
    app.run(port=5003)  # Run on port 5003
