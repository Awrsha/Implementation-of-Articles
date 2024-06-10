# Generating New Data with Conditional Deep Convolutional GAN

In many classification tasks, the final model may not achieve sufficient accuracy due to limitations in the training data, such as a limited number of samples or an imbalanced distribution of classes. To mitigate these limitations, we can leverage Generative Adversarial Networks (GANs) to generate new samples and partially address the data constraints. In this exercise, we aim to generate new data using Conditional Deep Convolutional GANs (cDCGANs), which will be used for training and classification in another Convolutional GAN.

## Data Preparation

We start by loading and preprocessing the provided dataset, which consists of biomedical images showcasing the MedMNIST dataset.

## BreastMNIST Creation and Training

Using the provided dataset, we create the BreastMNIST dataset and train ResNet models for classification. Additionally, we evaluate the model performance on training and validation sets and visualize the training progress.

## Conditional DCGAN Architecture and Training

Next, we construct a Conditional DCGAN architecture, which incorporates both generator and discriminator networks. The architecture is similar to DCGANs but includes additional layers, especially in the generator and discriminator sections. We train the cDCGAN model and monitor the loss curves for both the generator and discriminator networks.

## Data Generation and Visualization

After training the cDCGAN, we generate new samples for each class and visualize a few of these generated samples to assess the quality of the generated data.

## Improving Generator Output

We discuss potential strategies for enhancing the quality of generated data and stabilizing the training process of the cDCGAN model.

## Classification with Generated Data

Finally, we combine the generated data with the original training data to create a new dataset with balanced class distributions. We repeat the training process with this new dataset and compare the results with those obtained in the initial training phase.

This exercise provides hands-on experience in using cDCGANs for generating new data samples, which can be beneficial for enhancing the robustness and generalization capability of machine learning models.

## References

- [Conditional Deep Convolutional GANs](References/cDCGAN.pdf)