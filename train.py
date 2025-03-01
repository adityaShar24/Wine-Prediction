import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#Moder Dir
model_dir = "./app/model/"
os.makedirs(model_dir, exist_ok=True)


# Load dataset
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")

print(f'Red df- {red.head()}')
print(f'White df - {white.head()}')
print(f'White columns - {white.columns}')
print(f'Red columns - {red.columns}')

# Add labels: 1 for red wine, 0 for white wine
red["type"] = 1
white["type"] = 0

# Combine datasets
wines = pd.concat([red, white], ignore_index=True)

# Prepare input and output
X = wines.iloc[:, :-1].values  # Features (all except last column)
y = wines["type"].values  # Labels


# # Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,)

X_train = X_train[:, :11]


# Build model
model = Sequential([
    Dense(12, activation="relu", input_shape=(11,)),
    Dense(9, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=8, verbose=1)

# Save model
model.save(os.path.join(model_dir, "wine_model.h5"))
print(f"Model saved at {os.path.join(model_dir, 'wine_model.h5')}")