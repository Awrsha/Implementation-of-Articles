## Overview

This project involves using an Auto-Encoder to solve a classification problem with the MNIST dataset. The project is structured in three main parts: working with the MNIST dataset, designing and training an Auto-Encoder, and using the Auto-Encoder for classification. We will be using libraries such as PyTorch or TensorFlow/Keras for this purpose.

## 1. Working with the MNIST Dataset

### Data Preparation

First, we will familiarize ourselves with the MNIST dataset, which contains images of handwritten digits (0-9). We will use the following steps to prepare the dataset:

1. **Loading the Dataset**: Use the `torchvision.datasets.MNIST` function to load the dataset.
2. **Exploring the Dataset**: Plot the number of samples per class for both training and testing sets.
3. **Random Sample Visualization**: Visualize random samples from the dataset.
4. **Normalization**: Normalize the data for further processing.

### Visualizing Data Distribution

Plot the number of samples per class for both training and testing datasets.

## 2. Designing and Training the Auto-Encoder

### Auto-Encoder Architecture

The Auto-Encoder network will have two main parts: the Encoder and the Decoder.

#### Encoder Architecture

| Layer | Description             |
|-------|-------------------------|
| Input | 784 (flattened 28x28 image) |
| FC1   | 500 units               |
| FC2   | Optional layer, 100 units |
| FC3   | 30 units (bottleneck layer) |

#### Decoder Architecture

| Layer | Description             |
|-------|-------------------------|
| Input | 30 units (bottleneck layer) |
| FC1   | 100 units               |
| FC2   | Optional layer, 500 units |
| Output| 784 units (flattened 28x28 image) |

### Training the Model

1. **Model Training**: Train the Auto-Encoder on the MNIST dataset.
2. **Loss Calculation**: Use the mean squared error as the loss function.
3. **Plotting Loss**: Plot the training and validation loss over epochs.

### Example Architecture Table

**Encoder**:
- Input: 784
- FC1: 500
- FC2: 100 (Optional)
- Output: 30

**Decoder**:
- Input: 30
- FC1: 100
- FC2: 500 (Optional)
- Output: 784

## 3. Classification Using the Auto-Encoder

### Feature Extraction

1. **Using Encoder Output**: After training the Auto-Encoder, use the output of the Encoder as features for classification.
2. **Separating Encoder**: Use the Encoder part separately to extract the 30-dimensional feature vector from the input images.

### Classification Network

1. **Network Architecture**: Design a simple classifier with two hidden layers.
2. **Training**: Train the classifier using the features extracted by the Encoder.
3. **Evaluation**: Evaluate the classifier on the test dataset.

### Visualization

1. **Plot Metrics**: Plot accuracy, validation accuracy, loss, and validation loss over epochs.
2. **Confusion Matrix**: Plot the confusion matrix for the test dataset and interpret the results.

### Example Classifier Architecture

- **Input Layer**: 30 units (features from Encoder)
- **Hidden Layer 1**: Number of units (to be defined)
- **Hidden Layer 2**: Number of units (to be defined)
- **Output Layer**: 10 units (one for each class)

### Reporting Results

1. **Accuracy**: Report the accuracy on the test dataset.
2. **Confusion Matrix**: Plot and analyze the confusion matrix to understand the performance of the classifier.

## Conclusion

This project provides a comprehensive understanding of using Auto-Encoders for feature extraction and subsequent classification tasks. By following the steps outlined above, we can effectively preprocess the MNIST dataset, train an Auto-Encoder, and use it to build a robust classifier.

## References

- Include any references or resources used for the project, such as research papers, articles, or documentation.