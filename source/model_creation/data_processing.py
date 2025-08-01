import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import os
import numpy as np
import re

DATASET = "/home/nico/Desktop/UNTWISTER-Framework/dataset"


class LSTMDataset(Dataset):
    def __init__(self, dataframe, feature_columns, target_column):
        self.data = dataframe
        self.features = feature_columns
        self.target = target_column
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the features (X) and target (y)
        x = self.data.iloc[idx][self.features].values
        y = self.data.iloc[idx][self.target]
        
        # Convert to PyTorch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y
    
def create_padded_sequences(df, feature_cols, target_col, seq_length, padding_value=0):
    """
    Creates padded sequences of feature data from a DataFrame and labels based on the target column.

    Args:
    - df (pandas.DataFrame): The DataFrame containing the time series data.
    - feature_cols (list of str): List of column names to be used as features.
    - target_col (str): Column name containing the target labels (0 or 1 for normal or anomalous).
    - seq_length (int): The length of the sequences to generate. Sequences shorter than this will be padded.
    - padding_value (int/float): Value used to pad the sequences (default is 0).

    Returns:
    - sequences (numpy.ndarray): An array containing the padded feature sequences.
    - labels (numpy.ndarray): An array containing the corresponding labels.
    """
    # Print the feature_cols you're trying to access
    print("Feature columns:", feature_cols)
    sequences = []
    labels = []
    data = df[feature_cols].values  # Extract feature data
    targets = df[target_col].values  # Extract target labels

    # Loop through the data and create sequences
    for i in range(len(data) - seq_length + 1):
        # Extract the sequence of features (of length `seq_length`)
        sequence = data[i:i + seq_length]

        # If the sequence is shorter than seq_length, pad it with the padding_value
        if len(sequence) < seq_length:
            padding = np.full((seq_length - len(sequence), len(feature_cols)), padding_value)
            sequence = np.vstack([padding, sequence])

        # Append the padded sequence
        sequences.append(sequence)
        # Append the label of the last item in the sequence (target for anomaly detection)
        labels.append(targets[i + seq_length - 1])

    # Convert sequences and labels to numpy arrays
    return np.array(sequences), np.array(labels)
    
def load_data(attack_mapping):
    all_results = []

    for dirpath, dirnames, filenames in os.walk(DATASET):
        #print(f"Processing file: {dirpath}")
        for filename in filenames:
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                df = pd.read_csv(file_path, decimal='.', sep=',')
                
                # Get the attack name from the filename
                attack_name = filename.split('.')[0]  # Assuming the attack name is the first part of the filename                
                # Assign target value based on attack name
                if attack_name in attack_mapping:
                    ##############################################################################################################################
                    # not general part
                    if attack_name == 'lead_speed_attack_ramp':
                        df['target'] = 0  # Default for first row
                        df['attackParam'] = 0  # Default value
                        delete_indices = []
                        condition_triggered = False
                        for i in range(1, len(df)):
                            if df.loc[i, '{ramp_attack}.ramp_attackInstance.attackedParam'] != df.loc[i-1, '{lead}.leadInstance.speed']:
                                df.loc[i, 'target'] = attack_mapping[attack_name]
                                """ if not condition_triggered:
                                    attack_time = df.loc[i-1, 'time']  # Store first attack time """
                                condition_triggered = True
                                if i + 1 >= len(df): 
                                    df.loc[i, 'attackParam'] = ((df.loc[i, '{ramp_attack}.ramp_attackInstance.attackedParam'] - df.loc[i-1, '{lead}.leadInstance.speed']) - (df.loc[i-1, '{ramp_attack}.ramp_attackInstance.attackedParam'] - df.loc[i - 2, '{lead}.leadInstance.speed'])) / 0.01
                                elif df.loc[i + 1, '{ramp_attack}.ramp_attackInstance.attackedParam'] == df.loc[i, '{lead}.leadInstance.speed']:
                                    df.loc[i, 'attackParam'] = ((df.loc[i, '{ramp_attack}.ramp_attackInstance.attackedParam'] - df.loc[i-1, '{lead}.leadInstance.speed']) - (df.loc[i-1, '{ramp_attack}.ramp_attackInstance.attackedParam'] - df.loc[i - 2, '{lead}.leadInstance.speed'])) / 0.01
                                #time_diff = df.loc[i, 'time'] - attack_time
                                #if time_diff != 0:
                                else:
                                    df.loc[i, 'attackParam'] = ((df.loc[i+1, '{ramp_attack}.ramp_attackInstance.attackedParam'] - df.loc[i, '{lead}.leadInstance.speed']) - (df.loc[i, '{ramp_attack}.ramp_attackInstance.attackedParam'] - df.loc[i - 1, '{lead}.leadInstance.speed'])) / 0.01
                            else:
                                if condition_triggered:
                                    delete_indices.append(i) 
                                df.loc[i, 'target'] = 0
                        df.drop(delete_indices, inplace=True) # Elimino ciò che succede dopo l'attacco perchè tanto non mi interessa
                        df['{lead}.leadInstance.speed'] = df['{ramp_attack}.ramp_attackInstance.attackedParam'] # Assuming the attack output is the lead speed
                        df = df.drop(columns=['{ramp_attack}.ramp_attackInstance.attackedParam']) # Drop the attack output column
                    
                    if attack_name == 'lead_speed_attack_plus_constant':
                        df['target'] = 0  # Default for first row
                        df['attackParam'] = 0  # Default value
                        delete_indices = []
                        condition_triggered = False
                        for i in range(1, len(df)):
                            if df.loc[i, '{plus_constant_attack}.plus_constant_attackInstance.attackedParam'] != df.loc[i-1, '{lead}.leadInstance.speed']:
                                df.loc[i, 'target'] = attack_mapping[attack_name]
                                condition_triggered = True
                                df.loc[i, 'attackParam'] = df.loc[i, '{plus_constant_attack}.plus_constant_attackInstance.attackedParam'] - df.loc[i - 1, '{lead}.leadInstance.speed']
                            else:
                                if condition_triggered:
                                    delete_indices.append(i)
                                df.loc[i, 'target'] = 0
                        df.drop(delete_indices, inplace=True)
                        df['{lead}.leadInstance.speed'] = df['{plus_constant_attack}.plus_constant_attackInstance.attackedParam'] # Assuming the attack output is the lead speed
                        df = df.drop(columns=['{plus_constant_attack}.plus_constant_attackInstance.attackedParam']) # Drop the attack output column
                    
                    if attack_name == 'normal':
                        df['target'] = attack_mapping[attack_name]
                        df['attackParam'] = 0  # Default value
                    ##############################################################################################################################
                else:
                    df['target'] = -1  # Default case for unknown attacks

                all_results.append(df)

    combined_data = pd.concat(all_results, ignore_index=True) # curly braces are not allowed in column names
    combined_data.to_csv("../combined_data.csv", index=False)
    print("combined data stored")
    return combined_data

def preprocess_data(df, config, type):
    """
    Preprocessa i dati per l'MLP senza applicare lo scaling (usiamo BatchNorm invece).

    Parameters:
    - df: DataFrame originale
    - config: Dizionario con la configurazione delle feature

    Returns:
    - DataFrame con feature preprocessate e target
    """
    df_clean = df.dropna().copy()

    # Rinominiamo dinamicamente le colonne per rimuovere caratteri speciali
    rename_mapping = {col: col.replace("{", "").replace("}", "") for col in df_clean.columns}
    df_clean.rename(columns=rename_mapping, inplace=True)

    # Calcolo delle feature istantanee
    for feature, details in config["instant"].items():
        expression = details["expression"]

        for col in sorted(df_clean.columns, key=len, reverse=True):  
            expression = re.sub(rf'\b{re.escape(col)}\b', f'df_clean["{col}"]', expression)

        #print(f"Evaluating feature: {feature}")
        #print(f"Expression: {expression}")

        try:
            df_clean[feature] = eval(expression)  
            #print(f"Feature '{feature}' calcolata con successo!")
        except Exception as e:
            print(f"Errore nella valutazione della feature {feature}: {e}")
            raise

    # Calcolo delle feature rolling
    for feature, details in config["rolling"].items():
        expression = details["expression"]
        rolling_window = details["rolling_window"]

        for col in df_clean.columns:
            expression = expression.replace(col, f'df_clean["{col}"]')

        expression = expression.replace('window_size', str(rolling_window))

        try:
            df_clean[feature] = eval(expression)  
        except Exception as e:
            print(f"Errore nella valutazione della feature {feature}: {e}")
            raise

    # Eliminazione righe con NaN dopo il rolling
    df_clean = df_clean.dropna()

    # Definiamo solo le feature da restituire, senza scaling
    features = list(config["instant"].keys()) + list(config["rolling"].keys())

    return df_clean[features + ['target']]
