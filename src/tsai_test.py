"""TSAI"""

import sys
import os
import logging
import pandas as pd
from tsai.all import Path  # type: ignore

# Ensure project root is on sys.path so absolute imports work when running this module as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preparation import ts_to_pd_dataframe, pd_dataframe_to_csv

def save_ts_to_csv():
    # DSID = ["HouseholdPowerConsumption1", "HouseholdPowerConsumption2", "BeijingPM10Quality", "BeijingPM25Quality"],
    DSID = "HouseholdPowerConsumption1"
    SPLIT_DATA = False  # type: ignore
    FULL_TARGET_DIR = "./data/" + DSID
    PATH = "./data/downloaded"

    for split in ["TEST", "TRAIN"]:
        fname: Path = Path(PATH) / f"{DSID}_{split}.ts"
        try:
            if split == "TRAIN":
                X_train, y_train = ts_to_pd_dataframe(fname)  # type: ignore
                if not isinstance(X_train, pd.DataFrame):
                    break
                pd_dataframe_to_csv(X_train, FULL_TARGET_DIR+"_"+split+".csv")
            else:
                X_valid, y_valid = ts_to_pd_dataframe(fname) # type: ignore
                if not isinstance(X_valid, pd.DataFrame):
                    break
                pd_dataframe_to_csv(X_valid, FULL_TARGET_DIR+"_"+split+".csv")
        except ValueError as inst:
            logging.error(
                "Cannot create numpy arrays for %s dataset. Error: %s", DSID, inst
            )

def save_multidim_to_csv():
    # "BIDMC32RR" uses raw CSV because of windowing in TS data
    DSID = "BIDMC32"
    FULL_TARGET_DIR = "./data/" + DSID
    PATH = "./data/from_csv"
    df = pd.read_csv(PATH+"/"+DSID+".csv") # type: ignore
    print(df.head())
    headers: list[str] = df.columns.values.tolist()
    for header in headers:
        df[header].to_csv(
            f"{FULL_TARGET_DIR}_dim{header}.csv",
            index=True,
            header=False
        )

# np.save(f'{full_tgt_dir}/X_train.npy', X_train)
# np.save(f'{full_tgt_dir}/y_train.npy', y_train)
# np.save(f'{full_tgt_dir}/X_valid.npy', X_valid)
# np.save(f'{full_tgt_dir}/y_valid.npy', y_valid)
# np.save(f'{full_tgt_dir}/X.npy', concat(X_train, X_valid))
# np.save(f'{full_tgt_dir}/y.npy', concat(y_train, y_valid))
# PATH = Path('./models/Regression.pkl')

# X, y, splits = get_regression_data(DSID, split_data=False)
# print(X.shape, y.shape)
# check_data(X,y,splits, False)
# new_X = np.reshape(X[0][1], -1)

# tfms  = [None, [TSRegression()]]
# batch_tfms = TSStandardize(by_sample=True, by_var=True)
# dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=128)
# learn = ts_learner(dls, InceptionTime, metrics=[mae, rmse])
# learn.fit_one_cycle(50, 1e-2)

# PATH.parent.mkdir(parents=True, exist_ok=True)
# learn.export(PATH)
# learn = load_learner(PATH, cpu=False)
# raw_preds, target, preds = learn.get_X_preds(X[splits[1]])
# np.save('test.npy', preds)
# print('predictions saved, preds.shape')

# print(raw_preds)
# print(target)
# print(preds)
