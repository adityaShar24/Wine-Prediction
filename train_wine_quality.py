import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Model Directory
model_dir = "./app/model/"
os.makedirs(model_dir, exist_ok=True)

# Load datasets
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")

# Add wine type as a feature (Optional)
red["type"] = 1  # Red Wine = 1
white["type"] = 0  # White Wine = 0

# Combine datasets
wines = pd.concat([red, white], ignore_index=True)

# Prepare input and output
X = wines.drop(columns=["quality"])  # Features (All except quality)
y = wines["quality"].values  # Target variable (Wine Quality Score)

# Normalize features (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Scale training data

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Build regression model
quality_model = Sequential([
    Dense(12, activation="relu", input_shape=(X_train.shape[1],)),  # Ensure input shape is 11
    Dense(9, activation="relu"),
    Dense(1, activation="linear")  # Linear activation for regression
])

# Compile model
quality_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])

# Train model
quality_model.fit(X_train, y_train, epochs=5, batch_size=8, verbose=1)

# Save model
model_path = os.path.join(model_dir, "wine_quality_model.h5")
quality_model.save(model_path)
print(f"Wine Quality Model saved at {model_path}")
