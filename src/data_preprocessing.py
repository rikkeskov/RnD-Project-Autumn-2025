"""
The datasets for used in this project contain missing values.
It is the believe that SHRINK will perform the best with no missing values.
Therefore the missing values will need to be filled before.
SHRINK assumes piecewise linear relationships between the data points in the univariate time series.
Therefore the three methods for filling missing data used in this project (for now) is.
 - Mean
 - Linear interpolation (by pandas library)
 - K-nearest neighbours imputation

Source (mean): medium.com/@team_77175/data-preprocessing-in-ml-handling-missing-time-series-data-ecb7ad1a5da4
Source (linear interpolation): geeksforgeeks.org/data-analysis/handling-missing-values-machine-learning/
Source (knn): scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
"""
import sys
import os

from enum import Enum
from sklearn.impute import KNNImputer
from typing import Any

import pandas as pd
import numpy as np

from tsai.all import *  # type: ignore

# Ensure project root is on sys.path so absolute imports work when running this module as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shrink.constants import DATA_PATH

class PreprocessingType(Enum):
    MEAN = 0
    INTERPOLATION = 1
    KNN = 2

def missing_value_count(df: pd.DataFrame) -> int:
    count: int = int(df.isnull().sum()[1]) # type: ignore
    print(f"There is {count} missing values.")
    return count

def fill_by_mean(filename: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(DATA_PATH + filename, sep=",", header=None, index_col=0)
    nan_count: int = missing_value_count(df)
    if nan_count > 0:
        mean_imputed_df: pd.DataFrame = df.copy() # type: ignore
        mean: float = mean_imputed_df.iloc[1].mean() # type: ignore
        mean_imputed_df.fillna(mean, inplace=True) # type: ignore
        if isinstance(mean_imputed_df, pd.DataFrame):
            return mean_imputed_df
    print("Return type after imputation is not pandas df.")
    return df

def fill_by_interpolation(filename: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(DATA_PATH + filename, sep=",", header=None, index_col=0)
    nan_count: int = missing_value_count(df)
    if nan_count > 0:
        interpolated_df: pd.DataFrame = df.copy() # type: ignore
        interpolated_df = interpolated_df.interpolate(method='linear') # type: ignore
        if isinstance(interpolated_df, pd.DataFrame):
            return interpolated_df
    print("Return type after imputation is not pandas df.")
    return df

def fill_by_knn(filename: str, k: int = 5) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(DATA_PATH + filename, sep=",", header=None, index_col=0)
    nan_count: int = missing_value_count(df)
    if nan_count > 0:
        knn_imputed_df: pd.DataFrame = df.copy() # type: ignore
        imputer = KNNImputer(missing_values=np.nan, n_neighbors=k, weights="distance", copy=False)
        imputer.fit_transform(knn_imputed_df) # type: ignore
        if isinstance(knn_imputed_df, pd.DataFrame):
            return knn_imputed_df
    print("Return type after imputation is not pandas df.")
    return df

def preprocessing_and_save_as_csv(filename: str, preprocessing_type: PreprocessingType, output_path: str = "data/preprocessed", k: int = 5) -> str:
    if preprocessing_type == PreprocessingType.MEAN:
        df = fill_by_mean(filename)
    elif preprocessing_type == PreprocessingType.INTERPOLATION:
        df = fill_by_interpolation(filename)
    elif preprocessing_type == PreprocessingType.KNN:
        df = fill_by_knn(filename, k)
    else:
        print(f"Could not understand preprocessing type. {filename} not preprocessed.")
        exit()
    full_out_path: str = output_path + filename.split(".")[0] + f"_{preprocessing_type.name}.csv"
    df.to_csv(full_out_path, header=False)
    # pd_dataframe_to_csv(df, full_out_path)
    return full_out_path

def generate_X_y(file: str, window_length: int, horizon: int, stride: int | None) -> tuple[Any]:
    # load as pandas df
    df = pd.read_csv(f"{DATA_PATH}{file}", sep=",", header=None, index_col=0) # type: ignore
    print(f"Dataset name: {file}. Dataset shape: {df.shape}.")
    try:
        X, y = SlidingWindow(window_length, horizon=horizon, stride=stride)(df) # type: ignore
    except Exception as e:
        print(f"Could not generate X, y pair from Pandas dataframe: {e}")
        exit()
    print(f"X shape: {X.shape}, y shape: {y.shape} with first 10 values being: {y[:10]}") # type: ignore
    return X, y # type: ignore

def make_ts_splits(X: Any, valid_pct: float=0.2) -> tuple[list[int], list[int]]:
    n = len(X)
    cut = int(n * (1 - valid_pct))

    train_idx = list(range(cut))
    valid_idx = list(range(cut, n))

    return train_idx, valid_idx

if __name__ == "__main__":
    df_interp = fill_by_interpolation("/BeijingPM10Quality_TEST_dim0.csv")
    print(df_interp.iloc[121043]) # type: ignore
    missing_value_count(df_interp)
    df_mean = fill_by_mean("/BeijingPM10Quality_TEST_dim0.csv")
    print(df_mean.iloc[121043]) # type: ignore
    missing_value_count(df_mean)
    df_knn = fill_by_knn("/BeijingPM10Quality_TEST_dim0.csv")
    print(df_knn.iloc[121043]) # type: ignore
    missing_value_count(df_knn)
