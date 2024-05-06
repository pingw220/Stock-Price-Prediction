# Apple Stock Price Prediction Using LSTM

## Overview
This project uses a Long Short-Term Memory (LSTM) neural network to predict Apple's stock prices. The code is implemented using PyTorch and leverages historical stock price data from Yahoo Finance. The LSTM model is trained to understand the patterns in historical stock price movements and to make future predictions.

## Features
- **Data Fetching**: Downloads historical stock data for Apple (AAPL) from Yahoo Finance.
- **Data Preprocessing**: Scales the stock prices using MinMaxScaler for better neural network performance.
- **LSTM Model**: Uses a custom LSTM neural network for sequential data processing.
- **Prediction**: Makes future price predictions based on the learned patterns in the data.
- **Visualization**: Plots the actual and predicted prices for evaluation.

## Dependencies
To run this project, you will need the following libraries:
- Python 3.6+
- torch
- yfinance
- numpy
- pandas
- matplotlib
- scikit-learn

You can install these packages via pip:
```
pip install torch yfinance numpy pandas matplotlib scikit-learn
```

## Usage
To run this project, simply clone the repository, navigate to the directory containing the script, and execute the script using Python:
```
python stock_price_prediction.py
```

## Structure
- `stock_price_prediction.py`: Contains the main script with data fetching, preprocessing, model training, and evaluation.

## Model Details
- **Input Size**: 1 (price at a time)
- **Hidden Layer Size**: 100 (number of LSTM units)
- **Output Size**: 1 (predicted price)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)

## Output
The program outputs two plots:
1. A plot of Apple's historical closing stock prices.
2. A comparison plot showing actual vs. predicted stock prices based on the test dataset.

## Results
The model's performance is evaluated using the Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). These metrics are printed at the end of the script's execution.
