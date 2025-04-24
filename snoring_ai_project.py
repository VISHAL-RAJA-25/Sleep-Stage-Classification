import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ✅ 1. Extract MFCC from a single audio file
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)  # Transpose then average to fix the shape

# ✅ 2. Path to your dataset
data_path = "C:/AI project/Snoring Dataset"

# ✅ 3. Extract features and labels
features = []
labels = []

for label in os.listdir(data_path):
    class_path = os.path.join(data_path, label)
    if os.path.isdir(class_path):
        for file in os.listdir(class_path):
            if file.endswith('.wav'):
                file_path = os.path.join(class_path, file)
                mfcc = extract_mfcc(file_path)
                features.append(mfcc)
                labels.append(label)

# ✅ 4. Convert to NumPy arrays
X = np.array(features)
y = np.array(labels)

# ✅ 5. Encode the labels (strings to numbers)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ✅ 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ✅ 7. Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ 8. Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully!")
print("✅ Accuracy on test data:", accuracy)
print("✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))
# Step 8: Save the model
import joblib
joblib.dump(model, "snoring_sleep_model.pkl")