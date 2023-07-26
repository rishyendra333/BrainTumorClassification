import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np

!unzip /content/BrainTumor.zip

# Data Loading and Preprocessing
desired_width = 128
desired_height = 128

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

X_train = []
Y_train = []

# Load and preprocess images from the Training folder
for label in labels:
    folder_path = os.path.join('/content/BrainTumor/Training', label)
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        img = cv2.resize(img, (desired_width, desired_height))
        X_train.append(img)
        Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Convert class labels to integer indices and one-hot encode them
Y_train = np.array([labels.index(label) for label in Y_train])
Y_train = tf.keras.utils.to_categorical(Y_train)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=101)

# Model Definition
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(desired_width, desired_height, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

# Model Compilation and Training
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
epochs = 20
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

model.save('brain_tumor_model.h5')
