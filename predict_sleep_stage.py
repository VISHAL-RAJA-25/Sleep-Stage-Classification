import joblib
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Load the trained model
model = joblib.load("snoring_sleep_model.pkl")

# Define the possible sleep stage labels
labels = ['DeepSleep', 'LightSleep', 'REM', 'Awake']
encoder = LabelEncoder()
encoder.fit(labels)

# Function to extract MFCC features
def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        print(f"✅ MFCC extracted: shape = {mfcc.shape}")
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"❌ Failed to extract MFCC: {e}")
        return None

# Test audio path
test_audio_path = "test_audio.wav"

# Check if file exists and predict
if os.path.exists(test_audio_path):
    mfcc_features = extract_mfcc(test_audio_path)
    if mfcc_features is not None:
        mfcc_features = mfcc_features.reshape(1, -1)
        prediction = model.predict(mfcc_features)
        predicted_label = encoder.inverse_transform(prediction)[0]
        print(f"🧠 Predicted Sleep Stage: {predicted_label}")
else:
    print("❌ Test audio file not found!")
