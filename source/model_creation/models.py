import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader 

# LSTM Model Definition
class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Binary classification

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]  # Use the last hidden state
        out = self.fc(last_hidden_state)
        return out  # Raw logits (no sigmoid)

""" # Definizione della MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])  # BatchNorm dopo il primo layer
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])  # BatchNorm dopo il secondo layer
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))  # BatchNorm prima di ReLU
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  # No BatchNorm sull'output (CrossEntropyLoss lo gestisce)
        return x """

#Definizione della rete neurale MLP
class MLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        hidden = 128 # Numero di neuroni nei livelli nascosti

        # Definizione dei livelli condivisi tra classificazione e regressione
        self.shared_layer = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Testa per la classificazione
        self.classification_head = nn.Linear(hidden, num_classes)
        # Testa per la regressione
        self.regression_head = nn.Linear(hidden, 1)
    
    def forward(self, x):
        shared_output = self.shared_layer(x) # Propagazione nei livelli condivisi
        class_output = self.classification_head(shared_output)  # Output della classificazione
        reg_output = self.regression_head(shared_output)  # Output della regressione
        return class_output, reg_output  # Restituisce entrambi gli output

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.int64)
        
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y
    
    def __len__(self):
        return self.labels.shape[0]