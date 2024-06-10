# Cryptocurrency Price Prediction using Recurrent Neural Networks

This task involves familiarizing with recurrent neural network models, specifically GRU and LSTM, through the implementation of research papers titled:

- **A Deep Learning-based Cryptocurrency Price Prediction Scheme for Financial Institutions**

The objective is to predict the prices of Monero and Litecoin cryptocurrencies for three-day and seven-day time frames. If parameters are not specified in the papers, arbitrary values can be chosen, but it should be clearly mentioned in the report.

## Model Descriptions

Firstly, provide an overview of GRU and LSTM models, discussing their advantages and disadvantages. Then, explain the architectural differences presented in the papers for GRU and LSTM models, along with their functioning.

## Dataset Preparation and Preprocessing

Utilize the Investing.com dataset mentioned in the paper for preprocessing. This dataset includes cryptocurrency price data. Extract and preprocess the data as described in the paper.

## Model Training

Implement the models proposed in the paper, both the GRU and LSTM models, and train them on the preprocessed dataset. Evaluate the models using Mean Absolute Error (MAE) and Mean Squared Error (MSE), and plot the error curves.

## Evaluation and Results Analysis

Evaluate the models based on MSE for both cryptocurrencies and different time frames. Calculate additional metrics like Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and Mean Absolute Error (MAE). Present the results in tabular form and analyze the performance.