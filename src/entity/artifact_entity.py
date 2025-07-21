from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    x_train: np.ndarray
    y_train : np.ndarray
    x_test: np.ndarray
    y_test : np.ndarray

@dataclass
class MetricArtifact:
    mse:float
    mae:float
    rmse:float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
    metric_artifact:MetricArtifact

@dataclass
class ModelPpoularTrainerArtifact:
    models_popular_file_path:str 
  