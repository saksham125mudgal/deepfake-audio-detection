import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from src.model import create_model 
import os

def train_model(config):
    """Train the PyTorch LSTM model"""
    # Load processed data
    X_train = np.load('data/processed/train/features.npy')
    y_train = np.load('data/processed/train/labels.npy')
    X_val = np.load('data/processed/dev/features.npy')
    y_val = np.load('data/processed/dev/labels.npy')
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, 
                            batch_size=config['training_params']['batch_size'],
                            shuffle=True)
    
    # Create model
    input_size = X_train.shape[2]
    model = create_model(input_size, config)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), 
                         lr=config['model_params']['learning_rate'])
    
    # Training loop
    best_accuracy = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(config['training_params']['epochs']):
        model.train()
        running_loss = 0.0
        
        # Training phase
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            val_preds = (val_outputs > 0.5).float()
            val_accuracy = (val_preds == y_val_tensor.to(device)).float().mean()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy.item())
        
        print(f"Epoch {epoch+1}/{config['training_params']['epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'models/lstm_model.pth')
            print("Saved best model!")
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_accuracy': val_accuracies
    }
    torch.save(history, 'models/training_history.pth')
    
    return model, history