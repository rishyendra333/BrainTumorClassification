import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Define the class labels
class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary tumor']

# Load the trained model
model = load_model('path/to/your/brain_tumor_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)
    return img

def predict_image_class(image_path):
    try:
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        class_index = np.argmax(prediction[0])
        class_label = class_labels[class_index]
        return class_label
    except Exception as e:
        print(f"Error predicting the image: {e}")
        return None

if __name__ == "__main__":
    image_path = '/path/to/your/input_image.jpg'  # Replace with the path to your input image
    predicted_class = predict_image_class(image_path)

    if predicted_class is not None:
        print(f"The predicted class for the input image is: {predicted_class}")
    else:
        print("Prediction failed.")
