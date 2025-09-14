import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, config):
    """Evaluate the PyTorch model"""
    # Load test data
    X_test = np.load('data/processed/eval/features.npy')
    y_test = np.load('data/processed/eval/labels.npy')
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Predictions
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        y_pred_proba = test_outputs.cpu().numpy()
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    report = classification_report(y_test, y_pred, 
                                 target_names=['spoof', 'bonafide'])
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['spoof', 'bonafide'],
                yticklabels=['spoof', 'bonafide'])
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')
    
    print("Classification Report:")
    print(report)
    
    return report, cm