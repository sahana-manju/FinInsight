from src.exception import MyException
from src.logger import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
from keras.models import load_model

import os
from pathlib import Path



class PredictionPipeline:
    def __init__(self, portfolio_value : float,stock_options: list,weights: list):
        self.portfolio_value = portfolio_value
        self.stock_options = stock_options
        self.weights = weights

    def get_latest_artifact_dir(self,base_path="artifact"):
        """Return path to the most recent folder in artifact/"""
        base = Path(base_path)
        folders = [f for f in base.iterdir() if f.is_dir()]
        if not folders:
            raise FileNotFoundError("No folders found in artifact directory.")
        
        latest = max(folders, key=os.path.getmtime)
        return latest

    def load_model_and_scaler(self,ticker, base_path="artifact"):
        """
        Loads the latest model and scaler for a given ticker from the artifact directory.
        Returns: (model, scaler)
        """
        latest_dir = self.get_latest_artifact_dir(base_path)
        
        model_path = latest_dir / "saved_models" / f"{ticker}_model.keras"
        scaler_path = latest_dir / "scaled_models" / f"{ticker}_scaler.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler


    def forecast_next_30(self,model, scaler, last_30):
        """
        Predicts next 30 days using recursive LSTM forecasting.
        Returns: list of 30 unscaled predictions.
        """
        last_30.columns = ['Close'] 

        scaled = scaler.transform(last_30)
        input_window = scaled.copy()
        predictions = []

        for _ in range(30):
            input_seq = input_window.reshape(1, 30, 1)
            next_scaled = model.predict(input_seq, verbose=0)
            next_unscaled = scaler.inverse_transform(next_scaled)[0][0]
            predictions.append(next_unscaled)
            input_window = np.append(input_window, next_scaled, axis=0)[1:]

        return predictions
    
    def run_pipeline(self):
        stock_data = yf.download(self.stock_options,period='30d')['Close']
        logging.info("Downloaded previous dataset for all companies")
        logging.info("Forecasting started")
        forecast_dict = {}
        for ticker in self.stock_options:
            model, scaler = self.load_model_and_scaler(ticker)
            
            last_30 = stock_data[[ticker]]

            forecast = self.forecast_next_30(model, scaler, last_30)
            forecast_dict[ticker] = forecast
        logging.info("Forecasting complete saving the dataframe")
        forecast_df = pd.DataFrame(forecast_dict)
        forecast_df.index = range(1, 31)  # Day numbers
        logging.info("Completed process")
        return forecast_df







