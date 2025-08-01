from flask import Flask, request, jsonify
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import os
import zipfile
import torch.nn.functional as F
import joblib
import requests
from sklearn.model_selection import train_test_split
from dataset_generation import dataset_generation
from data_processing import load_data, preprocess_data, LSTMDataset, create_padded_sequences
from models import LSTMAnomalyDetector, MLP, MyDataset
from training import train_lstm, train_mlp
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

app = Flask(__name__)

SAVE_ZIP = "/home/nico/Desktop/UNTWISTER-Framework/Untwister&model_creation/saved_models/models_and_scalers.zip"
dataset_path = "../combined_data.csv"

def send_model_and_scaler(models: dict, scalers: dict, url: str, save_zip_path: str) -> bool:
    """
    Sends PyTorch models and their associated scalers as a ZIP archive to the specified URL,
    and saves the ZIP file locally.

    Args:
    - models (dict): A dictionary where keys are model names and values are PyTorch model objects.
    - scalers (dict): A dictionary where keys are scaler names and values are sklearn scaler objects.
    - url (str): The server URL to which the data will be sent.
    - save_zip_path (str): Path to save the ZIP file locally.

    Returns:
    - bool: True if the data was successfully sent, False otherwise.
    """
    # Prepare temporary directory to store the serialized models and scalers
    temp_dir = "temp_model_and_scaler_data"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Serialize and save the models and scalers to files
        for model_name, model in models.items():
            model_path = os.path.join(temp_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model {model_name} saved to {model_path}")

        for scaler_name, scaler in scalers.items():
            scaler_path = os.path.join(temp_dir, f"{scaler_name}.joblib")
            joblib.dump(scaler, scaler_path)
            print(f"Scaler {scaler_name} saved to {scaler_path}")
        
        # Create a ZIP file containing both models and scalers
        zip_filename = "models_and_scalers.zip"
        zip_filepath = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w') as zipf:
            # Add model and scaler files to the ZIP archive
            for file_name in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file_name)
                if file_name.endswith(".pth") or file_name.endswith(".joblib"):
                    zipf.write(file_path, os.path.basename(file_path))
            print(f"Created ZIP file: {zip_filepath}")

        # Save ZIP file locally to the specified path
        os.makedirs(os.path.dirname(save_zip_path), exist_ok=True)
        os.rename(zip_filepath, save_zip_path)
        print(f"Saved ZIP file locally at: {save_zip_path}")

        # Send the ZIP file to the server
        with open(save_zip_path, 'rb') as zip_file:
            files = {'file': (zip_filename, zip_file, 'application/zip')}
            response = requests.post(url, files=files)
            response.raise_for_status()  # Raise error if the response status code is not 2xx
            
            # Check the server response
            if response.status_code == 200:
                print("Successfully sent models and scalers.")
                return True
            else:
                print(f"Failed to send models and scalers. Server response: {response.status_code}")
                return False
    except Exception as e:
        print(f"Error occurred while sending models and scalers: {e}")
        return False
    finally:
        # Clean up temporary directory
        for file_name in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file_name)
            os.remove(file_path)
        os.rmdir(temp_dir)
        print(f"Cleaned up temporary files in {temp_dir}")

@app.route('/model_creation', methods=['POST'])
def model_creation():
    config_data = request.json
    
    if not all(key in config_data for key in ["dse", "attacks", "lstm_features", "mlp_features"]):        
        return jsonify({"status": "failure", "message": "Invalid configuration structure."}), 400
    
    #print(f"Model Creation Component received configuration: {config_data}")
    
    # Retrieve the DSE and COE configurations
    model_data = config_data["dse"]

    # start the DSE to generate the dataset
    #dataset_generation(model_data)
    """ if os.path.exists(dataset_path):
        print("Il dataset già esiste, lo carico...")
        dataset = pd.read_csv(dataset_path, nrows=100000)
    else: """
    #print("Il dataset non esiste, comincio a generare...")
    dataset = load_data(config_data.get("attacks", {})) 

    """ # estraggo le features
    df_mlp_preprocessed = preprocess_data(dataset, config_data.get("mlp_features", {}), 1)     

    # Separiamo features e target
    mlp_features = df_mlp_preprocessed.drop(columns=['target']).values
    mlp_targets = df_mlp_preprocessed['target'].values

    # Creiamo il train/test split (75/15)
    X_train, X_test, y_train, y_test = train_test_split(
        mlp_features, mlp_targets, test_size=0.15, random_state=42, stratify=mlp_targets)
    
    # Creiamo il train/validation split (75/15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    print("Training size: ", X_train.shape)
    print("Validation size: ", X_val.shape)
    print("Test size: ", X_test.shape)

    print("Training labels: ", np.bincount(y_train))
    print("Validation labels: ", np.bincount(y_val))
    print("Test labels: ", np.bincount(y_test))

    train_ds = MyDataset(X_train, y_train)
    val_ds = MyDataset(X_val, y_val)
    test_ds = MyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=32,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=32,
        shuffle=False,
    )

    def compute_accuracy(model, dataloader):
        model = model.eval()

        correct = 0.0
        total_examples = 0

        for idx, (features, labels) in enumerate(dataloader):
            with torch.inference_mode():
                logits = model(features)

            predictions = torch.argmax(logits, dim=1)

            compare = labels == predictions
            correct += torch.sum(compare)
            total_examples += len(compare)

        return correct / total_examples
    
    torch.manual_seed(1)
    model = MLP(num_features=6, num_classes=3)
    optimizer = torch.optim.adamw(model.parameters(), lr=0.05) #stochastic gradient descent

    num_epochs = 500
    for epoch in range(num_epochs):

        model = model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):

            logits = model(features)
            loss = F.cross_entropy(logits, labels) # Loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###LOGGING
            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f"  | Batch {batch_idx:03d}/{len(train_loader):03d}"
                  f"  | Train/Val Loss: {loss:.2f}")
            
        train_acc = compute_accuracy(model, train_loader)
        val_acc = compute_accuracy(model, val_loader)
        print(f"Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%")

    train_acc = compute_accuracy(model, train_loader)
    val_acc = compute_accuracy(model, val_loader)
    test_acc = compute_accuracy(model, test_loader)

    print(f"Train Acc {train_acc*100:.2f}%")
    print(f"Val Acc {val_acc*100:.2f}%")
    print(f"Test Acc {test_acc*100:.2f}%")
    
    # Store the model
    torch.save(model.state_dict(), "mlp_model.pth") """

    """ chunk_size = 100000  # Numero di righe da processare per volta
    epochs = 10
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Caricamento in chunks
    if os.path.exists(dataset_path):
        print("Il dataset esiste. Lo carico a chunk...")

        first_chunk = True
        mlp_model = None

        for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
            print("Processing a new chunk...")

            df_mlp_preprocessed = preprocess_data(chunk, config_data.get("mlp_features", {}), 1)

            # Separiamo features e target
            mlp_features = df_mlp_preprocessed.drop(columns=['target']).values
            mlp_targets = df_mlp_preprocessed['target'].values
            #print(f"mlp targets: {mlp_targets}")

            # Creiamo il train/validation split (80/20)
            X_train, X_val, y_train, y_val = train_test_split(mlp_features, mlp_targets, test_size=0.2, random_state=42)

            # Convertiamo in tensori
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

            # Creiamo il DataLoader per il training
            train_dataset = Tentorch.save(mlp_model.state_dict(), "mlp_model.pth")sorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            # **Creiamo il modello solo al primo chunk**
            if first_chunk:
                input_size = mlp_features.shape[1]  
                hidden_sizes = [16, 8]
                output_size = 3  # Numero di classi dinamico
                mlp_model = MLP(input_size, hidden_sizes, output_size).to(device)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
                first_chunk = False

            # **Allenamento del modello**
            for epoch in range(epochs):
                mlp_model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = mlp_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    # Calcoliamo l'accuracy
                    predictions = torch.argmax(outputs, dim=1)
                    correct += (predictions == targets).sum().item()
                    total += targets.size(0)

                train_accuracy = correct / total  # Accuracy sul training set
                # **Validazione**
                mlp_model.eval()
                correct_val = 0
                total_val = 0
                with torch.no_grad():
                    val_outputs = mlp_model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)

                    val_predictions = torch.argmax(val_outputs, dim=1)
                    correct_val += (val_predictions == y_val_tensor).sum().item()
                    total_val += y_val_tensor.size(0)
                val_accuracy = correct_val / total_val  # Accuracy sul validation set

                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}")
        # **Salviamo il modello dopo aver processato tutti i chunk**
        torch.save(mlp_model.state_dict(), "mlp_model.pth")
        print("Modello salvato con successo.")
    else:
        print("Il dataset non esiste, comincio a generare...")
        df = load_data(config_data.get("attacks", {}))
    """
    """  # preprocess data for the LSTM training 
    df_lstm_preprocessed, lstm_scaler = preprocess_data(df, config_data.get("lstm_features", {}), 0)
    df_lstm_preprocessed.to_csv("../lstm_feature.csv", index=False) """
    
    '''
    # train the LSTM for detection 
    features = df_lstm_preprocessed.drop(columns=['target']).columns
    target = 'target'

    seq_length = 10
      # Print the DataFrame columns
    print("Columns in df:", df.columns)

    X, y = create_padded_sequences(df_lstm_preprocessed, features, target, seq_length)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Dataset and DataLoader
    class TimeSeriesDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Hyperparameters
    input_size = len(features)  # Number of features
    hidden_size = 64           # Number of LSTM units
    num_layers = 2             # Number of LSTM layers
    learning_rate = 0.001
    dropout = 0.2

    # Model, loss, and optimizer
    lstm_model = LSTMAnomalyDetector(input_size, hidden_size, num_layers, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss()  # Combines Sigmoid + Binary Cross-Entropy
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.to(device)

    # Train the LSTM
    train_lstm(lstm_model, train_loader, val_loader, criterion, optimizer, 10, device)    
    '''
    """    val_inputs, val_targets = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

    mlp_input_size = X_train.shape[1]
    mlp_hidden_sizes = [128, 64]
    mlp_output_size = len(np.unique(y_train))

    mlp_model = MLPModel(mlp_input_size, mlp_hidden_sizes, mlp_output_size)
    mlp_loss_fn = nn.CrossEntropyLoss()
    mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

    train_mlp(mlp_model, train_dataloader, mlp_loss_fn, mlp_optimizer, epochs=10,
              validation_data=(val_inputs, val_targets)) """

    # Save models and scalers and send ZIP
    # models = {'lstm_model': lstm_model, 'mlp_model': mlp_model}
    # scalers = {'lstm_scaler': lstm_scaler, 'mlp_scaler': mlp_scaler}
    #url = "http://localhost:5000/upload_zip"  # Platform manager's URL

    #success = send_model_and_scaler(models, scalers, url, SAVE_ZIP)
    if True:#success:
        return jsonify({"status": "success", "message": "Model creation and sending process completed."})
    else:
        return jsonify({"status": "failure", "message": "Error sending models and scalers."}), 500

if __name__ == "__main__":
    app.run(port=5004)
