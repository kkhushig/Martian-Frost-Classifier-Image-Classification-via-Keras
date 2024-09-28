# Martian Frost Classifier - Image Classification via Keras
This project aims to develop a sentiment analysis classifier using deep learning techniques, specifically focusing on the analysis of positive and negative reviews. The classifier utilizes various architectures, including Multi-Layer Perceptrons (MLP), and Convolutional Neural Networks (CNN) to accurately predict sentiment from textual data.

By employing Keras and Python, this project demonstrates the application of advanced Natural Language Processing (NLP) techniques in text classification, paving the way for further exploration in sentiment analysis and related fields. 

> This repository contains a Python implementation of an image classification project for DSCI 552 which is focused on detecting frost in images using Convolutional Neural Networks (CNN) and Transfer Learning techniques with pre-trained models. The goal is to create a robust classification system that can distinguish between images with frost and those without it, leveraging the power of deep learning for high accuracy.

## Project Overview

### Objective
The primary objective of this project is to develop an image classification model that can accurately identify frost in images. This is particularly useful in agricultural settings where frost can significantly impact crop yield.

### Approach
1. **Custom CNN Model:** A three-layer CNN is designed to learn features from the images directly, followed by a fully connected layer for classification.
2. **Transfer Learning:** Pre-trained models such as EfficientNetB0, ResNet50, and VGG16 are employed to leverage existing knowledge from large datasets, which allows for faster training and potentially better performance.
3. **Data Augmentation and Preprocessing:** The dataset is augmented to improve model robustness. Techniques such as random flips, brightness adjustments, and contrast changes are applied to enrich the training data.
4. **Model Evaluation:** The models are evaluated using metrics such as accuracy, precision, recall, and F1-score to determine their effectiveness.

## Implementation Details

### Technology Stack
1. **Programming Language:** Python
2. **Deep Learning Framework:** TensorFlow and Keras
3. **Libraries:** NumPy, Matplotlib, Scikit-learn, TQDM, Pillow

### Data Structure
The dataset should be organized as follows:

```
data/
├── train_source_images.txt
├── val_source_images.txt
├── test_source_images.txt
├── <subdir_1>/
│   ├── tiles/
│   │   ├── <class_1>/
│   │   ├── <class_2>/
│   └── labels/
│       ├── <class_1>.txt
│       └── <class_2>.txt
└── <subdir_2>/
    ├── tiles/
    └── labels/
```

## Loading and Preprocessing Data
1. **Data Loading:** The script includes a function to load image file paths and labels from text files. The images are loaded in RGB format and resized to 299x299 pixels to match the input requirements of pre-trained models.
2. **Data Augmentation:** Augmentation techniques are applied during the training phase to enhance model performance. This includes random flips, contrast adjustments, and brightness variations.

## Model Architecture
### Custom CNN Model

The architecture of the custom CNN model consists of:
1. **Convolutional Layers:** Three convolutional layers with increasing filter sizes (32, 64, 128) and ReLU activation functions, followed by batch normalization and max pooling layers for downsampling.
2. **Dense Layer:** A fully connected layer with 256 neurons, ReLU activation, and dropout for regularization.
3. **Output Layer:** A softmax output layer with two neurons (binary classification).

### Transfer Learning Models

Pre-trained models are integrated into the architecture with the following steps:
1. Load the model without the top layers (classification layer).
2. Freeze the base model to retain the learned weights.
3. Add custom layers (flattening, dense, batch normalization, and dropout) to adapt the model for the specific classification task.

## Model Training

1. **Training Process:** The models are trained for a minimum of 20 epochs with an early stopping mechanism based on validation loss.
2. **Loss Function:** Sparse categorical cross-entropy is used due to the binary classification nature of the task.
3. **Optimizer:** The Adam optimizer is employed for efficient weight updates.

## Evaluation
After training, the models' performance is evaluated on training, validation, and test sets using classification reports that include accuracy, precision, recall, and F1-score.

## Results
The results of the training and evaluation of the models are logged, showing their performance metrics. The project includes visualizations of training and validation loss and accuracy over epochs.

## Observations
The overall observation shows that the accuracy is in the range of 52% to 56%. When comparing CNN-MLP to the transfer learning models, they have similar accuracy. 
*(CNN-MLP (56%) has higher testing accuracy than EffcientNetB0 (54%) and VGG16 (55%) though has similar testing accuracy to ResNet50 (56%).)*
In terms of **Precision, Recall and F-1 Score, CNN-MLP** has a higher precision and recall for Class 1 whereas for the Transfer Learning models, the values are more balanced. Thus, 
  1. CNN-MLP might be preferred in cases where minimizing false positives for Class 1 is crucial.
  2. EffcientNetB0, ResNet50, and VGG16 might be preferred in cases where balanced performance across both classes is more important.
