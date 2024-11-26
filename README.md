# Google Stock Price Prediction using RNN, LSTM, and GRU

## Project Overview

This project explores the application of Recurrent Neural Networks (RNNs) for predicting Google stock prices using historical stock data. The project aims to compare the performance of three different RNN architectures: Vanilla RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU), to identify the most effective model for stock price prediction.

The dataset used in this project consists of daily stock price data for Google, sourced from Kaggle. The task involves predicting future stock prices based on historical data, which is a key challenge in financial forecasting.

---

## Dataset

The dataset used in this project is **Google_Stock_Price_Train.csv**, which contains daily stock prices for Google over a decadal period. The dataset includes the following columns:

- **Date**: The date of the stock price entry
- **Open**: Opening price of the stock on that day
- **High**: The highest price of the stock during the day
- **Low**: The lowest price of the stock during the day
- **Close**: The closing price of the stock on that day
- **Volume**: The total number of shares traded on that day

---

## Objectives

- **Data Preprocessing**: Clean the data, handle missing values, and normalize the stock prices using Min-Max scaling.
- **Model Building**: Build three different models to predict the stock prices:
  1. **Vanilla RNN**: A basic recurrent neural network model.
  2. **LSTM**: A Long Short-Term Memory model that captures long-term dependencies.
  3. **GRU**: A Gated Recurrent Unit model that is similar to LSTM but simpler and more efficient.
- **Model Evaluation**: Use the Root Mean Squared Error (RMSE) to evaluate the performance of each model. Compare the predictions of the models and choose the best one based on performance.
- **Data Splitting**: The dataset is split into training and testing datasets, with 80% used for training and 20% used for testing.

---

## Features

- **Data Preprocessing**: Removing commas in the 'Close' column, converting string values to float, and normalizing the data.
- **Model Comparison**: Evaluation of three RNN architectures: Vanilla RNN, LSTM, and GRU.
- **Visualizations**: Plots comparing the training vs testing data, and actual vs predicted stock prices.
- **Metrics**: RMSE and other common metrics to evaluate model performance.

---

## Technologies Used

- **Python**: Programming language used to implement the models.
- **Pandas**: Data manipulation and cleaning.
- **NumPy**: Numerical operations.
- **Matplotlib**: Plotting library for visualizations.
- **TensorFlow/Keras**: Deep learning framework used for building and training the RNN, LSTM, and GRU models.
- **Scikit-learn**: Machine learning library for data scaling and evaluation metrics.

---

