# Playing Card Classification using Convolutional Neural Networks (CNNs)

This project focuses on classifying playing cards by leveraging deep learning, specifically Convolutional Neural Networks (CNNs). The model is trained on a dataset comprising various images of playing cards.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [References](#references)

## Introduction

This project develops a CNN-based model to classify playing cards into distinct categories. The model is trained on a dataset containing images of playing cards from various suits and decks, aiming to accurately identify the card depicted in a given image.

## Dataset Overview

The dataset consists of a diverse set of images of playing cards, featuring different suits, decks, and ranks. Images are preprocessed to ensure uniformity in size and format.

- Dataset Size: 7,794 images
- Classes: 53 (representing each card type)
- Image Dimensions: 224 x 224 pixels in RGB format
- Train-Validation-Test Split: 7,264 / 265 / 265 images

More details about the dataset can be found at [this link](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification).

## Model Architecture

The architecture includes a series of convolutional layers followed by fully connected layers. Below is a summary of the structure:

### Convolutional Layers:

- Two convolutional layers (`conv1` and `conv2`) extract features from the input images.
- ReLU activation functions (`relu1` and `relu2`) introduce non-linearity.
- Max-pooling layers (`pool1` and `pool2`) reduce the spatial dimensions and downsample feature maps.

### Fully Connected Layers:

- After the convolutional layers, the feature maps are flattened and passed through two fully connected layers (`fc1` and `fc2`).
- The first fully connected layer (`fc1`) contains 512 neurons with ReLU activation (`relu3`).
- The final fully connected layer (`fc2`) outputs logits for each class.

### Input and Output:

- The model accepts RGB input images.
- The output layer has `num_classes` neurons, where `num_classes` equals 53 (representing each playing card).

### Forward Pass:

- Input images (`x`) undergo convolutional operations, followed by ReLU activation and max-pooling.
- The resulting feature maps are flattened and passed through the fully connected layers, yielding class logits.

The `CardClassifierCNN` model leverages convolutional and fully connected layers to learn and predict playing card categories.

For a detailed explanation, refer to the Model Architecture section in the code.

## Training Process

The model is trained using the Adam optimizer and Cross-Entropy Loss function over multiple epochs. Early stopping is implemented to avoid overfitting, with performance metrics tracked on validation data.

Refer to the Training Process section in the code for more information.

## Model Evaluation

Post-training, the model is evaluated using a separate test set. Performance metrics such as accuracy, precision, recall, and F1-score are calculated. Additionally, predictions on sample test images are visualized for qualitative analysis.

For more details, see the Model Evaluation section in the code.

## Usage

To run the model for inference, follow these steps:

1. Install the required dependencies listed in the Dependencies section.
2. Clone the repository.
3. Download the dataset and place it in the correct directory.
4. Run the provided scripts or execute the code in your development environment.

## Dependencies

Ensure the following dependencies are installed:

- Python (v3.9)
- PyTorch (v2.1.2)
- Matplotlib
- NumPy (v1.26.3)
- scikit-learn

## Contributing

Contributions are welcome! Feel free to open issues, submit pull requests, or provide feedback on the current implementation.

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)
- [Related Paper](https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier)