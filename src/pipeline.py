"""
Full pipeline of project.
Part 1: Preprocessing. Filling missing values
Part 2: Training model on training data. 
Part 3: Testing model on test data.
Part 4: Compressing test data with SHRINK.
Part 5: Testing model on SHRINK test data.
"""

import os
import sys

import pandas as pd

# Ensure project root is on sys.path so absolute imports work when running this module as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import missing_value_count, preprocessing_and_save_as_csv, PreprocessingType

FILENAME = "/BeijingPM10Quality_TEST_dim0.csv"
PREPROCESSING_TYPE = PreprocessingType.INTERPOLATION

if __name__ == "__main__":
    data_path = preprocessing_and_save_as_csv(FILENAME, PREPROCESSING_TYPE)
    df: pd.DataFrame = pd.read_csv(data_path, sep=",", header=None, index_col=0)
    nan_count: int = missing_value_count(df)
    if nan_count > 0:
        print("Error in preprocessing.")
        exit()
    
    # TODO: finish this file