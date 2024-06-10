## Overview

This project involves predicting car prices using a dataset provided for this purpose. The task includes data preprocessing, feature engineering, and training a Multi-Layer Perceptron (MLP) using TensorFlow/Keras or PyTorch. The project is divided into two main parts: data preprocessing and model training.

## 1. Data Preprocessing

### Objectives

The goal of this section is to familiarize yourself with loading, cleaning, and preprocessing data. Follow these steps:

1. **Loading the Data**:
   - Read the dataset from a CSV file using Pandas.
   - Display the number of missing values (`NaN`) for each column.

2. **Feature Engineering**:
   - Extract the company name from the `CarName` column and save it in a new column `CompanyName`.
   - Remove the columns `car_ID`, `CarName`, and `symboling`.
   - Correct any typos in the company names.
   - Convert categorical data to numerical data (e.g., fuel type can be encoded as 0 for gas and 1 for diesel). Use `pd.get_dummies` for this task.

3. **Correlation Analysis**:
   - Plot a correlation matrix to identify which feature has the highest correlation with the price.
   - Plot the distribution of prices and a scatter plot of the feature with the highest correlation against the price.

4. **Splitting and Scaling Data**:
   - Split the data into training and testing sets (85% training, 15% testing).
   - Use `MinMaxScaler` to scale the training and testing data separately to avoid data leakage.

## 2. Multi-Layer Perceptron (60 points)

### Objectives

This section focuses on training an MLP to predict car prices and analyzing the impact of different parameters. Follow these steps:

1. **Building MLP Models**:
   - Create three simple MLP models with 1, 2, and 3 hidden layers, respectively.

2. **Experimenting with Loss Functions and Optimizers**:
   - Experiment with different loss functions and optimizers for each model.

3. **Training and Evaluation**:
   - Train the models and plot the training and validation loss for each one.
   - Calculate and plot the R2 score for each model.
   - Select the best model based on the evaluation metrics.

4. **Final Model Testing**:
   - Use the best model to predict the prices for five randomly selected data points from the test set.
   - Compare the predicted prices with the actual prices and report the differences.

### Reporting Results

1. **Best Model Selection**:
   - Select the model with the highest R2 score as the best model.
   - Experiment with different loss functions and optimizers for the selected model.

2. **Predicting Prices**:
   - Randomly select five data points from the test set.
   - Predict their prices using the best model.
   - Compare the predicted prices with the actual prices and report the differences.

## Conclusion

This project provides a comprehensive approach to predicting car prices using an MLP. By following the steps outlined above, you will gain experience in data preprocessing, feature engineering, and training neural networks for regression tasks.

## References

- Include any references or resources used for the project, such as research papers, articles, or documentation.