"""TSAI"""

import sys
import os
import logging
import numpy as np  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
from tsai.all import *  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# Ensure project root is on sys.path so absolute imports work when running this module as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shrink.constants import DATA_PATH

full_tgt_dir = "./data/tsai"
STRIDE = None
HORIZON = 1

def generate_tsai_data(window_length: int, target_dir: str, train_file: str, test_file: str, generate_split: bool = False) -> None:
    # load as pandas df
    df_train = pd.read_csv(DATA_PATH + train_file, sep=",", header=None, index_col=0) # type: ignore
    # df_train.fillna(df_train.median()) # type: ignore
    df_test = pd.read_csv(DATA_PATH + test_file, sep=",", header=None, index_col=0) # type: ignore
    # df_test.fillna(df_test.median()) # type: ignore
    logging.info(f"Dataset name: {train_file}. Dataset shape: {df_train.shape}.")
    try:
        X_train, y_train = SlidingWindow(window_length, horizon=HORIZON, stride=STRIDE)(df_train)
        X_test, y_test = SlidingWindow(window_length, horizon=HORIZON, stride=STRIDE)(df_test)
    except Exception as e:
        logging.error(f"Could not generate X, y pair from Pandas dataframe: {e}")
    X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])
    # check_data(X, y, splits)
    tfms  = [None, [TSRegression()]]
    batch_tfms = TSStandardize(by_sample=True, by_var=True)
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=50)
    learn: Learner = ts_learner(dls, InceptionTime, metrics=[mae, rmse], cbs=ShowGraph())
    learn.lr_find()


def tsai_test():
    dsid = 'AppliancesEnergy' 
    X, y, splits = get_regression_data(dsid, split_data=False)
    print(X.shape)
    print(y.shape)
    print(y[:10])
    # check_data(X, y, splits)
    tfms  = [None, [TSRegression()]]
    batch_tfms = TSStandardize(by_sample=True, by_var=True)
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=128)
    dls.one_batch()
    print(f"dls.c {dls.c}")
    print(dls.show_batch())
    learn = ts_learner(dls, InceptionTime, metrics=[mae, rmse], cbs=ShowGraph())
    learn.lr_find()

if __name__ == "__main__":
    #generate_tsai_data(100, full_tgt_dir, "/HouseholdPowerConsumption1_TRAIN_dim0.csv", "/HouseholdPowerConsumption1_TEST_dim0.csv", True)
    tsai_test()