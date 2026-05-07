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

from tsai.all import *

# Ensure project root is on sys.path so absolute imports work when running this module as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import missing_value_count, preprocessing_and_save_as_csv, PreprocessingType, generate_X_y, make_ts_splits

PREPROCESSING_TYPE = PreprocessingType.INTERPOLATION

# Data set settings
DATASET = "BIDMC32"
DIMENSION = "II"
test_file = "/" + DATASET + "_TEST_dim" + DIMENSION + ".csv"
train_file = "/" + DATASET + "_TRAIN_dim" + DIMENSION + ".csv"
compressed_test_file = "/decompressed/" + DATASET + "_TEST_dim_" + DIMENSION + "_e0.01_eb0.02_decompressed.csv"
compressed_test_files = [f for f in os.listdir("data/decompressed") if DIMENSION in f]

# X y generation settings
WINDOW_LENGTH = 10
HORIZON = 1
STRIDE = None

# ML model settings
CHOSEN_MODEL = InceptionTime
BATCH_SIZE = 16

if __name__ == "__main__":
    train_df: pd.DataFrame = pd.read_csv("data" + train_file, sep=",", header=None, index_col=0)
    nan_count: int = missing_value_count(train_df)
    if nan_count > 0:
        train_file = preprocessing_and_save_as_csv(train_file, PREPROCESSING_TYPE)
        train_df: pd.DataFrame = pd.read_csv("data" + train_file, sep=",", header=None, index_col=0)
    nan_count: int = missing_value_count(train_df)
    if nan_count > 0:
        print("Error in preprocessing.")
        exit()
    X_train, y_train = generate_X_y(train_file, WINDOW_LENGTH, HORIZON, STRIDE)
    splits_train = make_ts_splits(X_train, valid_pct=0.2)

    tfms  = [None, [TSRegression()]]
    batch_tfms = TSStandardize(by_sample=True, by_var=True)
    dls = get_ts_dls(X_train, y_train, splits=splits_train, tfms=tfms, batch_tfms=batch_tfms, bs=BATCH_SIZE, shuffle_train=False, drop_last=False)
    assert dls.c == 1

    for i, b in enumerate(dls.valid):
        xb, yb = b
        assert xb.shape[0] > 0

    learn = ts_learner(dls, CHOSEN_MODEL, metrics=[mae, rmse])
    lr_finder = learn.lr_find()

    # find learning rate
    lrs = learn.recorder.lrs
    losses = learn.recorder.losses
    sma = 20
    derivatives = [0]*(sma+1)

    for i in range(1+sma, len(lrs)):
        d = (losses[i] - losses[i-sma]) / sma
        derivatives.append(d)

    derivatives = np.array(derivatives)

    best_idx = derivatives.argmin()
    best_lr = lrs[best_idx]

    # make model
    learn = ts_learner(dls, InceptionTime, metrics=[mae, rmse])
    learn.fit_one_cycle(50, best_lr)

    # find rmse value of model and save
    _, _, rmse_val = learn.validate()
    assert isinstance(rmse_val, float)
    path_model = Path(f"./models/regression_{DATASET}_dim{DIMENSION}_rmse{round(rmse_val, 5)}_{str(CHOSEN_MODEL.__name__)}.pkl")
    path_model.parent.mkdir(parents=True, exist_ok=True)
    learn.export(path_model)
    del learn

    # infer uncompressed test
    learn = load_learner(path_model, cpu=False)
    test_df: pd.DataFrame = pd.read_csv("data" + test_file, sep=",", header=None, index_col=0)
    nan_count: int = missing_value_count(test_df)
    if nan_count > 0:
        test_file = preprocessing_and_save_as_csv(test_file, PREPROCESSING_TYPE)
        test_df: pd.DataFrame = pd.read_csv("data" + test_file, sep=",", header=None, index_col=0)
    nan_count: int = missing_value_count(test_df)
    if nan_count > 0:
        print("Error in preprocessing.")
        exit()
    X_test, y_test = generate_X_y(test_file, WINDOW_LENGTH, HORIZON, STRIDE)
    probas, _, preds = learn.get_X_preds(X_test)
    rmse_test = skm.root_mean_squared_error(y_test, preds)

    results: list[tuple[str, float]] = []  # List to store (filename, rmse) tuples
    for file in compressed_test_files:
        X_comp, y_comp = generate_X_y("/decompressed/" + file, WINDOW_LENGTH, HORIZON, STRIDE)
        try:
            probas, _, preds = learn.get_X_preds(X_comp)
            rmse_test_comp = skm.root_mean_squared_error(y_comp, preds)
            results.append((file, rmse_test_comp))  # Add filename and RMSE to results
        except RuntimeError:
            print(f"too few data points in time series to generate one batch for file: {file}.")

    with open(f"data/tsai_results/{DATASET}_dim_{DIMENSION}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'rmse'])  # Header
        writer.writerow([train_file, rmse_val]) # Train rmse
        writer.writerow([test_file, rmse_test]) # Test uncompressed rmse
        writer.writerows(results)  # Write all rows
