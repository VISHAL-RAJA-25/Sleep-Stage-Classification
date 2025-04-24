# 🧠 AI-Powered Sleep Stage Classification Using Snoring Audio

This project detects a person's sleep stage (e.g., Awake, DeepSleep) by analyzing snoring audio using machine learning. It features a trained model and an interactive web interface built with Streamlit.

## 🔍 Problem Statement
Manual sleep monitoring is time-consuming and often expensive. This project aims to simplify and automate sleep stage detection through AI using only snoring audio.

## 🎯 Motivation
Sleep health is critical. By enabling affordable, contactless, and accessible analysis via audio, we empower people to track their sleep quality from home.

## 🏗️ Architecture Overview
- 📁 Dataset: Labeled snoring audio samples (Awake = 0, DeepSleep = 1)
- 🎚️ Preprocessing: Convert and extract MFCC (Mel-Frequency Cepstral Coefficients)
- 🧠 Model: RandomForestClassifier trained on extracted features
- 🌐 UI: Streamlit frontend for uploading and predicting sleep stages

## 🛠️ Features
- Upload `.wav` audio of snoring
- Display waveform and MFCC plots
- Predict sleep stage instantly
- Clean, modern UI with emoji animation 🎧

## 📦 Tech Stack
- Python, Librosa, Scikit-learn, Pydub
- Streamlit (for web UI)
- GitHub (version control)

## 🧪 Run Locally
```bash
git clone https://github.com/VISHAL-RAJA-25/Sleep-Stage-Classification
cd Sleep-Stage-Classification
streamlit run app.py
