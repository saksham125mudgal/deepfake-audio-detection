import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepfakeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(DeepfakeLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # Fully connected layers
        x = self.dropout(hidden)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x

def create_model(input_size, config):
    """Create and initialize the LSTM model"""
    model = DeepfakeLSTM(
        input_size=input_size,
        hidden_size=config['model_params']['hidden_size'],
        num_layers=config['model_params']['num_layers'],
        dropout=config['model_params']['dropout_rate']
    )
    
    return model