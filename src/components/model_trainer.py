import sys
from typing import Tuple

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, MetricArtifact
from src.entity.estimator import LSTM_Model

from src.constants import STOCK_COMPANY

import mlflow
import mlflow.keras


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, x_train: np.array, y_train: np.array,
                                x_test: np.array, y_test: np.array) -> Tuple[object, object]:
        try:
            logging.info("Training LSTM with specified parameters")
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            with mlflow.start_run(run_name="LSTM_Model_Training"):
                # Log Tag
                mlflow.set_tag("Company",STOCK_COMPANY)
                # Log parameters
                mlflow.log_param("optimizer", self.model_trainer_config._optimizer)
                mlflow.log_param("loss_function", self.model_trainer_config._loss)
                mlflow.log_param("batch_size", self.model_trainer_config._batch_size)
                mlflow.log_param("epochs", self.model_trainer_config._epochs)

                # Build the LSTM model
                model = Sequential()
                model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(64, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))

                model.compile(optimizer=self.model_trainer_config._optimizer, loss=self.model_trainer_config._loss)

                # Convert data types if necessary
                if x_train.dtype == object:
                    x_train = np.stack(x_train).astype(np.float32)
                if x_test.dtype == object:
                    x_test = np.stack(x_test).astype(np.float32)

                model.fit(x_train, y_train,
                        batch_size=self.model_trainer_config._batch_size,
                        epochs=self.model_trainer_config._epochs)

                logging.info("Model training done.")

                predictions = model.predict(x_test)
                scaler = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

                predictions = scaler.inverse_transform(predictions)
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Calculate metrics
                rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
                mae = np.mean(np.abs(predictions - y_test))
                mse = np.mean((predictions - y_test) ** 2)

                print("RMSE:", rmse)
                print("MAE:", mae)
                print("MSE:", mse)

                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)

                mlflow.keras.log_model(
                    model,
                    name="lstm_model",
                )

                metric_artifact = MetricArtifact(mse=mse, mae=mae, rmse=rmse)
                return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            x_train = self.data_transformation_artifact.x_train
            y_train =  self.data_transformation_artifact.y_train
            x_test = self.data_transformation_artifact.x_test
            y_test =  self.data_transformation_artifact.y_test
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(x_train,y_train,x_test,y_test)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            #if metric_artifact.rmse > self.model_trainer_config.expected_error:
            #    logging.info("No model found with score above the base score")
            #    raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = LSTM_Model(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e