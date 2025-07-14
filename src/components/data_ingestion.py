import os
import sys
import yfinance as yf

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.constants import STOCK_COMPANY, TARGET_COLUMN, LOOKBACK_PERIOD
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        

    def extract_realtime_data(self)->DataFrame:
        """
        Method Name :   extract_realtime_data
        Description :   This method exports data from yfinance api to dataframe
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from yfinace API")
            # Define the ticker
            ticker = STOCK_COMPANY


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

    def split_data_as_train_test(self,dataframe: DataFrame) ->None:
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
            test_set = dataframe[training_data_len-60:]
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise

    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            dataframe = self.extract_realtime_data()

            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise 