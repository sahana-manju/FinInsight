import os
from datetime import date


PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "Close"

STOCK_COMPANY = "AAPL"
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
TIMESTAMP = 60


