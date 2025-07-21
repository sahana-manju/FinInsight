import sys
from src.exception import MyException
from src.logger import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM
import pickle

from src.entity.config_entity import (TrainPopularConfig,ModelTrainerConfig
                                          )
                                          
from src.entity.artifact_entity import (ModelPpoularTrainerArtifact
                                           )
from src.constants import  LOOKBACK_PERIOD, TARGET_COLUMN, TIMESTAMP, POPULAR_TICKERS



class TrainPipeline_Popular:
    def __init__(self):
        self.popular_train_config = TrainPopularConfig()
        self.model_trainer_config = ModelTrainerConfig()
    
        
    def extract_realtime_data(self,ticker: str)->pd.DataFrame:
        """
        Method Name :   extract_realtime_data
        Description :   This method exports data from yfinance api to dataframe
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from yfinace API")
            # Define the ticker
            ticker = ticker


            # Download historical data using yfinance
            stock_data = yf.Ticker(ticker)
            df = stock_data.history(period=LOOKBACK_PERIOD)

            # Data Validation module
            if df.empty:
                error_message = f"No data returned for ticker '{ticker}' with lookback period '{LOOKBACK_PERIOD}'. " \
                                f"Check if the ticker or period is correct."
                logging.error(error_message)
                sys.exit(1)  # Exit with a non-zero code to indicate failure

            # Check if TARGET_COLUMN exists
            if TARGET_COLUMN not in df.columns:
                error_message = f"Target column '{TARGET_COLUMN}' not found in the dataset. Available columns: {df.columns.tolist()}"
                logging.error(error_message)
                sys.exit(1)  # Exit with a non-zero code to indicate failure
            df.reset_index(inplace=True)
            data = df.filter([TARGET_COLUMN])
            return data

        except Exception as e:
            raise

    def split_data_as_train_test(self,dataframe: pd.DataFrame) ->pd.DataFrame:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            training_data_len = int(len(dataframe) * 0.95)
            train_set = dataframe[:training_data_len]
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            return train_set
            
        except Exception as e:
            raise

    def get_data_transformer_object(self) -> MinMaxScaler:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            scaler = MinMaxScaler(feature_range=(0, 1))
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")
            return scaler


        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise

    def convert_to_timesteps(self,scaled_data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered convert_to_timesteps method of DataTransformation class")

        try:
            x_train = []
            y_train = []

            for i in range(TIMESTAMP,len(scaled_data)):
                x_train.append(scaled_data[i-TIMESTAMP:i,0])
                y_train.append(scaled_data[i,0])
            
            x_train, y_train = np.array(x_train), np.array(y_train)

            # Reshape the data (samples, timesteps, no of features)
            x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

            return x_train,y_train
            

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise

    def train_top_50_companies(self,x_train: np.array, y_train: np.array,ticker: str):
        try:
            logging.info("Training started")

            model = Sequential()
            model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            model.compile(optimizer=self.model_trainer_config._optimizer, loss=self.model_trainer_config._loss)

            # Convert data types if necessary
            if x_train.dtype == object:
                x_train = np.stack(x_train).astype(np.float32)

            model.fit(x_train, y_train,
                    batch_size=self.model_trainer_config._batch_size,
                    epochs=self.model_trainer_config._epochs)
            

            model.save(os.path.join(self.popular_train_config.models_popular_dir, f"{ticker}_model.keras"))


            logging.info("Model training done.")
            return

                

        except Exception as e:
            raise  
    

    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            for i in range(len(POPULAR_TICKERS)):
                logging.info(f"Training started for Company {i} - {POPULAR_TICKERS[i]}")
                df_stock = self.extract_realtime_data(POPULAR_TICKERS[i])
                train_set = self.split_data_as_train_test(df_stock)
                scaler_obj = self.get_data_transformer_object()
                scaled_data = scaler_obj.fit_transform(train_set)
                # Save the scaler
                scaler_dir = self.popular_train_config.scaled_models_file_path
                os.makedirs(scaler_dir, exist_ok=True)
                scaled_path = os.path.join(scaler_dir,f"{POPULAR_TICKERS[i]}_scaler.pkl")
                with open(scaled_path, "wb") as f:
                    pickle.dump(scaler_obj, f)
                x_train,y_train = self.convert_to_timesteps(scaled_data)
                self.train_top_50_companies(x_train, y_train,POPULAR_TICKERS[i])
                logging.info("Iteration Complete")
            
        

            
        except Exception as e:
            raise MyException(e, sys)