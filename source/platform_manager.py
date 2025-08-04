from flask import Flask, request, jsonify
import os
import json
import requests
from typing import Dict, Any
import time

app = Flask(__name__)

# Component URLs for sending configurations and start signals
COMPONENT_URLS: Dict[str, str] = {
    "data_collection": "http://localhost:5001/data_collection",
    "detection_classification": "http://localhost:5002/detection_classification",
    "attack_mitigation": "http://localhost:5003/attack_mitigation",
    "model_creation": "http://localhost:5004/model_creation",
    "reconfiguration": "http://localhost:5005/reconfiguration"
}

data_collection_url = "http://localhost:5001/start_work"
CONFIG_PARAMS_PATH = "./platform_manager/configuration_parameters.json"
ADVERSARY_MODEL_PATH = "./platform_manager/adversary_model.json"

model_file_path = "./platform_manager/saved_models/mlp_modelWithParam.pth"

def load_json_file(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)

def send_config(url: str, config_data: Dict[str, Any]) -> bool:
    """Send configuration data to a specified component."""
    try:
        response = requests.post(url, json=config_data)
        response.raise_for_status()

        print(f"Successfully sent config to {url}: {response.json()}")
        if response.json().get("status") == "success":
            return True
    except requests.exceptions.RequestException as e:
        print(f"Error while sending config to {url}: {e}")
    return False


def send_file(url: str, file_path: str) -> bool:
    """Send a file to a specified component."""
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(url, files={'file': f})
        response.raise_for_status()

        print(f"Successfully sent file to {url}: {response.json()}")
        if response.json().get("status") == "success":
            return True
    except requests.exceptions.RequestException as e:
        print(f"Error while sending file to {url}: {e}")
    return False


def send_start_signal() -> bool:
    """Send start signal to all components."""
    try:
        response = requests.post(data_collection_url, json={"message": "START_WORKING_PHASE"})
        response.raise_for_status()  # Raise an error for bad responses

        print(f"Successfully sent starting signal to {data_collection_url}: {response.json()}")
        return response.json().get("status") == "success"
    except requests.exceptions.RequestException as e:
        print(f"Error while sending start signal to data collection: {e}")
    return False


@app.route('/upload_zip', methods=['POST'])
def upload_zip():
    """Endpoint to receive the ZIP file from model_creation."""
    if 'file' not in request.files:
        return jsonify({"status": "failure", "message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "failure", "message": "No selected file"}), 400

    try:
        # Save the received ZIP file
        file.save(model_file_path)
        print(f"ZIP file saved at {model_file_path}")
        return jsonify({"status": "success", "message": "ZIP file received"})
    except Exception as e:
        print(f"Error handling ZIP file upload: {e}")
        return jsonify({"status": "failure", "message": f"Error processing file: {e}"}), 500


def configure_platform(config_data):
    """Handle configuration and start signal for the platform."""
    if "use_model_creation" not in config_data:
        return jsonify({"status": "failure", "message": "use_model_creation flag missing"}), 400

    use_model_creation = config_data["use_model_creation"]
    all_acks_received = True

    # Send configuration to each component
    for component, url in COMPONENT_URLS.items():
        # Skip model creation if `use_model_creation` is False
        if not use_model_creation and component == "model_creation":
            print(f"Skipping configuration for model_creation as use_model_creation is set to {use_model_creation}.")
            continue

        # Send JSON configuration
        component_config_data = config_data.get(component, {})
        if not send_config(url, component_config_data):
            print(f"Configuration failed for {component}. Stopping platform.")
            all_acks_received = False
            #break

    if use_model_creation:
        print("Waiting for AI model file from model_creation...")
        # Wait for the ZIP file from model_creation
        print("Still waiting for AI model file...")
        while not os.path.exists(model_file_path):
            time.sleep(1)

    print("Sending model file to detection_classification.")
    if not send_file(f"{COMPONENT_URLS['detection_classification']}/upload", model_file_path):
        print(f"File upload failed for detection_classification. Stopping platform.")
        all_acks_received = False

    if all_acks_received:
        print("All configurations complete. Sending start signal.")
        if not send_start_signal():
            print(f"Work phase failed. Stopping platform.")
            return jsonify({"status": "failure", "message": "Failed to start work phase."}), 500
        return jsonify({"status": "success", "message": "Platform configured and start signal sent."})
    else:
        print("Platform configuration unsuccessful due to one or more NACK responses. Not starting.")
        return jsonify({"status": "failure", "message": "Platform configuration unsuccessful."}), 500


""" @app.route('/load_config', methods=['POST'])
def load_config():
    #Load platform configuration from a JSON file.
    if 'file' not in request.files:
        return jsonify({"status": "failure", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "failure", "message": "No selected file"}), 400

    try:
        config_data = json.load(file)
        return configure_platform(config_data)  # Call the configure_platform function with the loaded config
    except Exception as e:
        return jsonify({"status": "failure", "message": f"Error processing file: {e}"}), 500 """

@app.route('/start_platform_from_files', methods=['POST'])
def start_platform_from_files():
    """Carica configuration_parameters.json e adversary_model.json e avvia la configurazione."""
    try:
        config_params = load_json_file(CONFIG_PARAMS_PATH)
        adversary_model = load_json_file(ADVERSARY_MODEL_PATH)

        # Unisci i due dizionari in uno solo
        full_config = {**config_params, **adversary_model}

        # Chiama il gestore principale
        return configure_platform(full_config)

    except Exception as e:
        return jsonify({"status": "failure", "message": f"Errore nel caricamento file JSON: {e}"}), 500

if __name__ == '__main__':
    app.run(port=5000)  # Run on port 5000 by default
