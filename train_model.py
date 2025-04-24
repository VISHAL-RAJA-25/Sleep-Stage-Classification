# train_model.py

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load features and labels
data = np.load("snoring_data.npz")
X, y = data["X"], data["y"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
with open("snoring_sleep_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as snoring_sleep_model.pkl")
