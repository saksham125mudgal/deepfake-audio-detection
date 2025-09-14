import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm

def load_cm_protocol(protocol_path):
    """Load CM protocol file for spoof detection labels"""
    data = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                filename = parts[1]
                attack_type = parts[3]
                label = 1 if attack_type == '-' else 0  # 1=real, 0=fake
                data.append({'filename': filename, 'label': label, 'attack_type': attack_type})
    return pd.DataFrame(data)

def extract_features(audio_path, config):
    """Extract MFCC features from audio file"""
    audio, sr = librosa.load(audio_path, sr=config['feature_params']['sr'])
    
    # Extract MFCCs and deltas
    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, 
        n_mfcc=config['feature_params']['n_mfcc']
    )
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)
    
    combined = np.vstack([mfccs, delta, delta2])
    
    # Pad/trim to fixed length
    max_len = config['feature_params']['max_len']
    if combined.shape[1] < max_len:
        padded = np.pad(
            combined, 
            ((0, 0), (0, max_len - combined.shape[1])),
            mode='constant',
            constant_values=0
        )
    else:
        padded = combined[:, :max_len]
    
    return padded.T  # Shape: (max_len, n_features)

def process_dataset(protocol_path, audio_dir, output_dir, config):
    """Process dataset and save features"""
    # Load protocol
    df = load_cm_protocol(protocol_path)
    
    features = []
    labels = []
    
    # Process each audio file
    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = os.path.join(audio_dir, "flac", f"{row['filename']}.flac")
        
        if os.path.exists(audio_path):
            feature = extract_features(audio_path, config)
            features.append(feature)
            labels.append(row['label'])
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'features.npy'), features)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    
    return features, labels