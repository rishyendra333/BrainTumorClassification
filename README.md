# Brain Tumor Classification

![Brain MRI](https://example.com/images/brain_mri.jpg)

Brain Tumor Classification is a deep learning project that aims to classify brain MRI images into four categories: glioma, meningioma, no tumor, and pituitary tumor. The project uses a Convolutional Neural Network (CNN) to achieve accurate classification results.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Brain Tumor Classification is an important task in medical imaging, as it helps in early detection and diagnosis of brain tumors. The project uses TensorFlow and Keras to build and train the CNN model for accurate classification.

## Dataset

The MRI data used for training and evaluation was collected from Kaggle's Brain Tumor Classification MRI dataset. The dataset includes brain MRI images belonging to four classes: glioma, meningioma, no tumor, and pituitary tumor.

The dataset was preprocessed and split into training and testing sets to train and evaluate the model. Data augmentation techniques such as random rotations, flips, and shifts were applied to increase the diversity of the training data and prevent overfitting.

## Model Architecture

The CNN model architecture is designed to effectively capture features from the brain MRI images and make accurate predictions. It consists of multiple convolutional layers with ReLU activation to extract important features, followed by max-pooling layers to reduce spatial dimensions and retain essential information.

The CNN is further connected to fully connected layers with ReLU activation to learn high-level representations from the extracted features. The final output layer uses the softmax activation function to produce probabilities for each class, enabling multi-class classification.

## Training and Evaluation

The model was trained on the preprocessed training data using the Adam optimizer and categorical cross-entropy loss function. During training, the model was validated on a separate portion of the training data to monitor its performance and prevent overfitting.

After training, the model was evaluated on the testing data to assess its generalization ability. The model achieved a validation accuracy of 92.31% and a training accuracy of 96.09%, demonstrating its effectiveness in classifying brain MRI images into the correct tumor categories.

## Results

The trained model demonstrates promising performance in classifying brain MRI images into the correct tumor categories. It effectively distinguishes between glioma, meningioma, no tumor, and pituitary tumor cases, which can be valuable for medical professionals in diagnosing brain tumors.

Further analysis and evaluation can be performed to explore the model's sensitivity, specificity, and other evaluation metrics to gain deeper insights into its performance and potential areas for improvement.

## Usage

To use the trained model for prediction, follow these steps:

1. Clone the repository:
`git clone https://github.com/your_username/Brain-Tumor-Classification.git`

2. Install the required dependencies:
`pip install -r requirements.txt`

3. Run the prediction script with your input image:
`python predict.py /path/to/your/input_image.jpg`
