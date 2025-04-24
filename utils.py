import librosa
import numpy as np
import pickle

def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfcc.T  # (frames, features)

def predict_stage(file_path, model_path="snoring_sleep_model.pkl"):
    features = extract_mfcc(file_path)
    features_mean = np.mean(features, axis=0).reshape(1, -1)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    prediction = model.predict(features_mean)[0]
    return "Awake" if prediction == 0 else "DeepSleep"
