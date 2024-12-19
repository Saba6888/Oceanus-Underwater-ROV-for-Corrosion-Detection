import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

def load_and_preprocess_images(directory, target_size=(224, 224)):
    images = []
    labels = []

    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, target_size)
            img = img / 255.0
            images.append(img)
            labels.append(label)

    return images, labels

# Example usage
dataset_path = "dataset"
X, y = load_and_preprocess_images(dataset_path)

for val in y:
    if (val == 'corroded'):
        y[y.index(val)] = 1
    else:
        y[y.index(val)] = 0

X = np.array(X)
y = np.array(y)

# Replace 'X' and 'y' with your actual data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Classification model
model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

model.save("corrosion_classification_model.h5")
result = model.predict(X_test)
binary_predictions = (result > 0.5).astype(int)
print("Confusion Matrix:\n", confusion_matrix(y_test, binary_predictions))
print("\nClassification Report:\n", classification_report(y_test, binary_predictions))

