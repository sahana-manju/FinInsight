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
    scaled_train_df : np.ndarray
    scaled_test_df : np.ndarray