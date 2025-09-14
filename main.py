import yaml
from src.data_processing import process_dataset
from src.train import train_model
from src.evaluate import evaluate_model
from src.model import create_model
import torch
import os
import config

def main():
    # Load configuration
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    os.makedirs('data/processed/train', exist_ok=True)
    os.makedirs('data/processed/dev', exist_ok=True)
    os.makedirs('data/processed/eval', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Process datasets 
    print("Processing training data...")
    process_dataset(
        'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
        'data/raw/ASVspoof2019_LA_train/',
        'data/processed/train/',
        config=config
    )

    print("Processing dev data.")
    process_dataset(
        'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt',
        'data/raw/ASVspoof2019_LA_train/',
        'data/processed/dev/',
        config=config
    )

    print("Processing evaluating data.")
    process_dataset(
        'data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt',
        'data/raw/ASVspoof2019_LA_train/',
        'data/processed/eval/',
        config=config
    )
    

    
    # Train model
    print("Training PyTorch model...")
    model, history = train_model(config)
    
    # Evaluate model
    print("Evaluating model...")
    report, cm = evaluate_model(model, config)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()