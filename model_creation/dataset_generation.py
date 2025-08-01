import os
from utils import save_json_temp, run_dse_algorithm, rename_results_csv, move_directories_to_data

TEMP_CONFIG_DIR = "/tmp/configurations"

# Function to process each model in the received JSON and start the geneation of the dataset
def dataset_generation(model_data):
    for model_name, model_configs in model_data.items():
        print(f"Processing model '{model_name}'")

        # Save temporary JSON files for DSE and COE configurations
        dse_temp_path = save_json_temp(model_configs["dse_conf"], f"{model_name}_dse.json")
        coe_temp_path = save_json_temp(model_configs["coe_conf"], f"{model_name}_coe.json")

        # Run the DSE algorithm
        run_dse_algorithm(dse_temp_path, coe_temp_path) #MAX 300 SIMULATIONS AT TIME!!!!!

        # Define results path based on the model name
        results_path = os.path.join(TEMP_CONFIG_DIR, model_name)

        # Ensure the results directory exists before renaming and moving
        if not os.path.exists(results_path):
            print(f"Warning: Expected results directory not found: {results_path}")
            continue  # Skip to the next model if results directory doesn't exist

        rename_results_csv(results_path, model_name)
        move_directories_to_data(results_path)