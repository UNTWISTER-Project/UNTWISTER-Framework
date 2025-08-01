import os
import socket
import json
import subprocess
import time
import shutil

COE_PORT = 8082
COE_JAR_PATH = "/home/nico/into-cps-projects/install_downloads/coe.jar"
TEMP_CONFIG_DIR = "/tmp/configurations"
ALGORITHM_SELECTOR_PATH = "/home/nico/Desktop/UNTWISTER-Framework/dse_scripts/Algorithm_selector.py"
PROJECT_PATH = "/home/nico/into-cps-projects"
DATA_DIR = "/home/nico/Desktop/UNTWISTER-Framework/dataset"

# Helper function to check if COE is running
def is_coe_active(host='localhost', port=COE_PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex((host, port))
        return result == 0  # Returns True if the port is open (active)

# Function to start the COE
def start_coe():
    try:
        print("Starting COE...")
        subprocess.Popen(['java', '-jar', COE_JAR_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Give COE time to start
        time.sleep(5)  # Adjust if necessary for longer startup times
        if is_coe_active():
            print("COE started successfully.")
        else:
            print("Failed to start COE.")
    except Exception as e:
        print(f"Error starting COE: {e}")

# Ensure temporary directory exists
os.makedirs(TEMP_CONFIG_DIR, exist_ok=True)

# Function to save JSON configuration to temporary files
def save_json_temp(data, filename):
    path = os.path.join(TEMP_CONFIG_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    return path

# Function to rename results.csv based on the COE name
def rename_results_csv(output_dir, coe_name):
    new_csv_name = f"{coe_name}.csv"
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file == 'results.csv':
                os.rename(os.path.join(root, file), os.path.join(root, new_csv_name))

# Function to run the INTO-CPS Algorithm_selector.py with temporary configuration files
def run_dse_algorithm(dse_json_path, coe_config_path, num_threads=1):
    # Check if COE is active and start it if not
    if not is_coe_active():
        start_coe()
    command = [
        'python3', ALGORITHM_SELECTOR_PATH,
        PROJECT_PATH,  # Required argument: Project path
        dse_json_path,  # Required argument: Path to .dse.json file
        coe_config_path,  # Required argument: Path to coe.json file
        '-t', str(num_threads),  # Optional: Set number of threads
        '-noHTML',  # Optional: Disable HTML output
        '-noCSV', # Optional: Disable CSV output
        '-d'  # Optional: Enable debug mode
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"DSE execution output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error during DSE execution: {e.stderr}")

def make_unique_name(base_name, directory):
    counter = 1
    new_name = base_name
    while os.path.exists(os.path.join(directory, new_name)):
        new_name = f"{base_name}_{counter}"
        counter += 1
    return new_name

def move_directories_to_data(results_path, data_dir=DATA_DIR):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(results_path):
        print(f"Warning: Results path does not exist: {results_path}")
        return  # Skip if the results path doesn't exist
    for item in os.listdir(results_path):
        item_path = os.path.join(results_path, item)
        if os.path.isdir(item_path) and item != os.path.basename(data_dir):
            destination_path = os.path.join(data_dir, make_unique_name(item, data_dir))
            shutil.move(item_path, destination_path)