import os
from datetime import date


PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "Close"

STOCK_COMPANY = "MSFT"
FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = f"{STOCK_COMPANY}_train.csv"
TEST_FILE_NAME: str = f"{STOCK_COMPANY}_test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
LOOKBACK_PERIOD = '10y'


AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "Stock-Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.20

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
PREPROCESSING_OBJECT_FILE_NAME: str = "scaler.pkl"
TIMESTAMP = 30

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
OPTIMIZER: str = "adam"
LOSS: str = "mean_squared_error"
BATCH_SIZE: int= 1
EPOCHS: int = 1

"""
POPULAR COMPANIES
"""
POPULAR_TICKERS = [
    'AAPL', 'MSFT'
    #'TSLA', 'NVDA', 'GOOGL', 'AMZN', 'META',
    # 'JNJ', 'JPM', 'V', 'MA', 'UNH', 'PG', 'HD', 'BAC', 'DIS', 
    # 'XOM', 'CVX', 'PFE', 'KO', 'PEP', 'WMT', 'CSCO', 'NFLX',
    # 'NKE', 'LLY', 'CRM', 'ABBV', 'T', 'MCD', 'ADBE', 'INTC',
    # 'CMCSA', 'AMD', 'BA', 'COST', 'WFC', 'GE', 'TMO', 'QCOM',
    # 'TXN', 'BMY', 'NEE', 'SBUX', 'HON', 'PYPL', 'LIN', 'AVGO',
    # 'LOW', 'UPS'
]
MODELS_POPULAR_DIR_NAME: str = "saved_models"
MODELS_POPULAR_FILE_NAME: str = "model.keras"
SCALED_POPULAR_DIR_NAME: str = "scaled_models"



