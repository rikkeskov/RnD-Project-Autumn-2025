"""TSAI"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from tsai.all import Path, SlidingWindow, test_eq  # type: ignore

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

def generate_tsai_data(window_length: int, target_dir: str, file: str, generate_split: bool = False) -> None:
    # load as pandas df
    df = pd.read_csv(DATA_PATH + file, sep=",", header=None, index_col=0) # type: ignore
    print(df.head())
    logging.info(f"Dataset name: {file}. Dataset shape: {df.shape}.")
    try:
        X, y = SlidingWindow(window_length)(df)
    except Exception as e:
        logging.error(f"Could not generate X, y pair from Pandas dataframe: {e}")
    print(X.shape)
    print(y.shape)

# def generate_tsai_model(tsai_data_folder: str) -> None:
#     np.save(f'{full_tgt_dir}/X_train.npy', X_train)
#     np.save(f'{full_tgt_dir}/y_train.npy', y_train)
#     np.save(f'{full_tgt_dir}/X_valid.npy', X_valid)
#     np.save(f'{full_tgt_dir}/y_valid.npy', y_valid)
#     np.save(f'{full_tgt_dir}/X.npy', concat(X_train, X_valid))
#     np.save(f'{full_tgt_dir}/y.npy', concat(y_train, y_valid))
#     PATH = Path('./models/Regression.pkl')

#     X, y, splits = get_regression_data(DSID, split_data=False)
#     print(X.shape, y.shape)
#     check_data(X,y,splits, False)
#     new_X = np.reshape(X[0][1], -1)

#     tfms  = [None, [TSRegression()]]
#     batch_tfms = TSStandardize(by_sample=True, by_var=True)
#     dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=128)
#     learn = ts_learner(dls, InceptionTime, metrics=[mae, rmse])
#     learn.fit_one_cycle(50, 1e-2)

#     PATH.parent.mkdir(parents=True, exist_ok=True)
#     learn.export(PATH)
#     learn = load_learner(PATH, cpu=False)
#     raw_preds, target, preds = learn.get_X_preds(X[splits[1]])
#     np.save('test.npy', preds)
#     print('predictions saved, preds.shape')

#     print(raw_preds)
#     print(target)
#     print(preds)

if __name__ == "__main__":
    generate_tsai_data(10, full_tgt_dir, "/HouseholdPowerConsumption1_TRAIN_dim0.csv", True)