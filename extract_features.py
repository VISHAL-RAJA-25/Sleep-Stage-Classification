# extract_features.py

import os
import numpy as np
import librosa

dataset_path = "Snoring Dataset"
features = []
labels = []

for label_folder in ["0", "1"]:
    label = int(label_folder)
    folder_path = os.path.join(dataset_path, label_folder)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            features.append(mfcc_mean)
            labels.append(label)

X = np.array(features)
y = np.array(labels)

# Save the dataset
with open("snoring_data.npz", "wb") as f:
    np.savez(f, X=X, y=y)

print("✅ Feature extraction complete! Saved as snoring_data.npz")
