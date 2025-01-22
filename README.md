# Gold Price Forecasting with LSTM-CNN Models

## Overview

This repository contains a comprehensive implementation of various machine learning and deep learning models for predicting gold prices. By leveraging the strengths of LSTM (Long Short-Term Memory) networks and CNNs (Convolutional Neural Networks), this project provides a robust framework for financial time-series forecasting.

## Key Features

- ARIMA Model: A statistical model for analyzing and forecasting time-series data.
- Support Vector Regression (SVR): A machine learning model for non-linear regression analysis.
- CNN Model: A deep learning model for extracting spatial features from the data.
- LSTM Model: A neural network designed for handling sequential data and capturing temporal dependencies.
### Hybrid Models
- CNN-LSTM: Combines CNN for spatial feature extraction with LSTM for temporal sequence learning.
- LSTM-CNN: A reversed hybrid approach prioritizing sequential learning before spatial feature extraction.
- LSTM-Attention-CNN: Incorporates an attention mechanism for enhanced feature learning.

## Dataset

The dataset includes daily gold price records. Preprocessing steps such as normalization (log-scale transformation) and data splitting (train-test ratio of 80:20) were applied to prepare the data for model training.

## Implementation Details

1. Data Preprocessing:
- Logarithmic transformation to stabilize variance.
- Scaling using MinMaxScaler for normalization.
- Time-series feature creation with a sliding window approach.
2. Model Architectures:
- Each model was trained for 100 epochs with a batch size of 32.
- Adam optimizer was used for all deep learning models.
- Evaluation metrics include Mean Absolute Error (MAE) and Mean Squared Error (MSE).
3. Visualization:
Predicted vs. actual values are visualized for each model.
Line plots provide insights into model performance over time.

## Results

<p align="center">
  <img src="https://github.com/SepehrNorouzi7/LSTM-CNN-GoldForecast/blob/main/screenshots/ARIMA.png" alt="Image 1" width="40%" />
  <img src="https://github.com/SepehrNorouzi7/LSTM-CNN-GoldForecast/blob/main/screenshots/SVR.png" alt="Image 2" width="40%" />
</p>

<p align="center">
  <img src="https://github.com/SepehrNorouzi7/LSTM-CNN-GoldForecast/blob/main/screenshots/CNN.png" alt="Image 3" width="40%" />
  <img src="https://github.com/SepehrNorouzi7/LSTM-CNN-GoldForecast/blob/main/screenshots/LSTM.png" alt="Image 4" width="40%" />
</p>

<p align="center">
  <img src="https://github.com/SepehrNorouzi7/LSTM-CNN-GoldForecast/blob/main/screenshots/CNN-LSTM.png" alt="Image 3" width="30%" />
  <img src="https://github.com/SepehrNorouzi7/LSTM-CNN-GoldForecast/blob/main/screenshots/LSTM-CNN.png" alt="Image 4" width="30%" />
  <img src="https://github.com/SepehrNorouzi7/LSTM-CNN-GoldForecast/blob/main/screenshots/LSTM-Attention-CNN.png" alt="Image 5" width="30%" />
</p>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Mean Absolute Error (MAE)</th>
      <th>Mean Squared Error (MSE)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ARIMA</td>
      <td>0.031050</td>
      <td>0.001401</td>
    </tr>
    <tr>
      <td>SVR</td>
      <td>0.005056</td>
      <td>0.000041</td>
    </tr>
    <tr>
      <td>CNN</td>
      <td>0.005676</td>
      <td>0.000051</td>
    </tr>
    <tr>
      <td>LSTM</td>
      <td>0.005403</td>
      <td>0.000046</td>
    </tr>
    <tr>
      <td>CNN-LSTM</td>
      <td>0.005646</td>
      <td>0.000049</td>
    </tr>
    <tr>
      <td>LSTM-CNN</td>
      <td>0.005230</td>
      <td>0.000044</td>
    </tr>
    <tr>
      <td>LSTM-Attention-CNN</td>
      <td>0.005222</td>
      <td>0.000044</td>
    </tr>
  </tbody>
</table>

