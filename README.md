Price Predictor Software

A deep learning-powered tool for predicting high and low price trends of a security using Recurrent Neural Networks (RNN) and traditional technical analysis indicators.

This project combines machine learning, time-series preprocessing, and anomaly detection techniques to deliver reliable price trend forecasts.

Features

Recurrent Neural Network (RNN): Custom-built model using TensorFlow to capture temporal dependencies in price data.

Anomaly Detection: Utilises Bollinger Bands and Slope analysis to detect and correct anomalies in time-series data.

High/Low Price Forecasting: Achieved an accuracy of 74.60% on historical data.

Modular Design: Easily adaptable to other securities or datasets.

Tech Stack

Languages: Python

Libraries: TensorFlow, Scikit-Learn, Pandas, NumPy, Matplotlib

Techniques: RNN, Bollinger Bands, Slope Analysis, Time-Series Modelling

Project Workflow

Data Collection: Historical price data of the chosen security.

Preprocessing & Cleaning:

Missing value treatment

Anomaly detection & removal using Bollinger Bands and Slope-based filtering

Feature Engineering: Extracting key features like moving averages, volatility measures, and returns.

Model Building:

RNN architecture designed to predict next-period high and low prices.

Model Evaluation: Evaluated using accuracy, RMSE, and visual inspection of predicted vs. actual values.

Results

Accuracy: 74.60% on validation data

Performance: Captured significant upward/downward trends effectively.

Getting Started
Prerequisites

Install dependencies:

pip install tensorflow scikit-learn pandas numpy matplotlib

Usage

Clone the repo:

git clone https://github.com/your-username/price-predictor-rnn.git
cd price-predictor-rnn


Run training:

python train.py --data data/your_dataset.csv


Predict prices:

python predict.py --input recent_data.csv

Project Structure
├── data/               # Sample dataset
├── src/
│   ├── preprocess.py   # Data cleaning & anomaly detection
│   ├── model.py        # RNN architecture
│   ├── train.py        # Training script
│   ├── predict.py      # Inference script
├── results/            # Model outputs and plots
└── README.md

Future Improvements

Incorporate LSTM/GRU for enhanced sequence modelling

Add real-time data streaming and prediction

Implement a backtesting module for strategy evaluation
