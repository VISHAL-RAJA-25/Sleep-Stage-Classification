import streamlit as st
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils import predict_stage

st.set_page_config(page_title="Sleep Stage Predictor", layout="centered")

st.title("AI-Powered Sleep Stage Predictor")
st.markdown("Upload your snoring audio (.wav) to detect sleep stage")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    # Save file temporarily
    file_path = os.path.join("temp.wav")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load audio
    y, sr = librosa.load(file_path)

    # Show waveform
    st.subheader("Waveform")
    fig1, ax1 = plt.subplots()
    ax1.plot(np.linspace(0, len(y)/sr, num=len(y)), y)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Audio Waveform")
    st.pyplot(fig1)

    # Show MFCCs
    st.subheader("MFCC (Mel-Frequency Cepstral Coefficients)")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    fig2, ax2 = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax2)
    ax2.set_title("MFCCs")
    fig2.colorbar(img, ax=ax2)
    st.pyplot(fig2)

    # Predict sleep stage
    prediction = predict_stage(file_path)

    # Show result
    st.success(f"Predicted Sleep Stage: {prediction}")