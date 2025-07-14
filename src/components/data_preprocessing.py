import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.compose import ColumnTransformer


from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, TIMESTAMP
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise 

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

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

            # # Load schema configurations
            # logging.info("Cols loaded from schema.")

            # # Creating preprocessor pipeline
            # preprocessor = ColumnTransformer(
            #     transformers=[
            #         ("MinMaxScaler", min_max_scaler, [TARGET_COLUMN])
            #     ],
            #     remainder='passthrough'  # Leaves other columns as they are
            # )

            # # Wrapping everything in a single pipeline
            # final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            # logging.info("Final Pipeline Ready!!")
            # logging.info("Exited get_data_transformer_object method of DataTransformation class")
            # return final_pipeline

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


    def initiate_data_transformation(self):
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

    

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Timestep creation for Training-data")
            x_train,y_train = self.convert_to_timesteps(input_feature_train_arr)
            logging.info("Timestep creation for Testing-data")
            x_test,y_test = self.convert_to_timesteps(input_feature_test_arr)
            logging.info("Timestep creation done end to end to train-test df.")



            

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test
            )


        except Exception as e:
            raise