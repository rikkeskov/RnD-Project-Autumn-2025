"""
Full pipeline of project.
Part 1: Preprocessing. Filling missing values
Part 2: Training model on training data. 
Part 3: Testing model on test data.
Part 4: Compressing test data with SHRINK.
Part 5: Testing model on SHRINK test data.
Part 6: Save statistics in CSV.
"""

import os
import sys
import time

from tsai.all import * # type: ignore

# Ensure project root is on sys.path so absolute imports work when running this module as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import (
    PreprocessingType,
    generate_X_y,
    make_ts_splits,
    preprocessing
)
from shrink_test import TestSHRINK, BaseEpsilonCalculation


# Data set settings
PREPROCESSING_TYPE = PreprocessingType.INTERPOLATION
DATASET = "BIDMC32"
DIMENSION = "AVR"
test_file = "data/" + DATASET + "_TEST_dim" + DIMENSION + ".csv"
train_file = "data/" + DATASET + "_TRAIN_dim" + DIMENSION + ".csv"
SHRINK_RESULTS_FILE_1 = f"data/shrink_results/{DATASET}_TEST_dim{DIMENSION}_results.csv"
SHRINK_RESULTS_FILE_2 = f"data/shrink_results/{DATASET}_TEST_dim{DIMENSION}_{PREPROCESSING_TYPE.name}_results.csv"

# X y generation settings
WINDOW_LENGTH = 20
HORIZON = 1
STRIDE = None

# ML model settings
CHOSEN_MODEL = InceptionTime
BATCH_SIZE = 16
CYCLES = 10

def find_lr_helper_func(learn: Learner) -> float:
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
    return best_lr

if __name__ == "__main__":
    """
    Part 1: Preprocessing. Filling missing values in training set
    """
    preprocessing(train_file, PREPROCESSING_TYPE)
    X_train, y_train = generate_X_y(train_file, WINDOW_LENGTH, HORIZON, STRIDE) # type: ignore
    splits_train = make_ts_splits(X_train, valid_pct=0.2)

    """
    Part 2: Training model on training data. Generate X and y, generate validation set
    """

    tfms  = [None, [TSRegression()]] # type: ignore
    batch_tfms = TSStandardize(by_sample=True, by_var=True)
    dls = get_ts_dls(
        X_train, # X training data
        y_train, # type: ignore # y training data
        splits=splits_train, # train / validation split
        tfms=tfms, # dataset transformers
        batch_tfms=batch_tfms, # batch transformers
        bs=BATCH_SIZE, # size of training batch
        shuffle_train=False, # cannot shuffle data because time series
        drop_last=False
    )
    assert dls.c == 1 # check univariate

    for i, b in enumerate(dls.valid):
        xb, yb = b
        assert xb.shape[0] > 0

    learn = ts_learner(dls, CHOSEN_MODEL, metrics=[mae, rmse])
    lr_finder = learn.lr_find()
    best_lr = find_lr_helper_func(learn)

    training_start = time.perf_counter()
    learn.fit_one_cycle(CYCLES, best_lr)
    training_time = time.perf_counter() - training_start

    # find rmse value of model and save
    _, _, rmse_val = learn.validate() # type: ignore
    assert isinstance(rmse_val, float)
    path_model = Path(f"./{DATASET}_dim{DIMENSION}_rmse{round(rmse_val, 5)}_{str(CHOSEN_MODEL.__name__)}.pkl")
    path_model.parent.mkdir(parents=True, exist_ok=True)
    learn.save(path_model)
    del learn

    """
    Part 3: Inferring model on uncompressed test data.
    """
    learn = ts_learner(
        dls,
        CHOSEN_MODEL,
        metrics=[mae, rmse]
    )
    learn.load(path_model, weights_only=False)
    print(f"Learner batch size is {learn.dls.bs}.")
    preprocessing(test_file, PREPROCESSING_TYPE)
    X_test, y_test = generate_X_y(test_file, WINDOW_LENGTH, HORIZON, STRIDE) # type: ignore
    print(f"Shape of X_test: {X_test.shape}, type: {type(X_test)} and length: {len(X_test)}.")
    print(f"Number of variables in learner: {learn.dls.vars} and length of data loader: {learn.dls.len}.")
    inference_start = time.perf_counter()
    probas, _, preds = learn.get_X_preds(X_test, bs=BATCH_SIZE)
    uncompressed_inference_time = time.perf_counter() - inference_start
    rmse_test = skm.root_mean_squared_error(y_test, preds) # type: ignore

    """
    Part 4: Compress and decompress test data if not already done.
    """
    if os.path.exists(SHRINK_RESULTS_FILE_1) or os.path.exists(SHRINK_RESULTS_FILE_2):
        print(
            f"SHRINK results already exist. Skipping SHRINK compression/decompression.")
    else:
        print("No SHRINK results found. Running SHRINK.")

        preprocessing(test_file, PREPROCESSING_TYPE)

        scaling_factors = [20, 25, 30, 35, 40]
        calculator = BaseEpsilonCalculation(test_file)
        base_epsilons = [
            calculator.compute_epsilon_base(f)
            for f in scaling_factors
        ]
        residual_epsilons = [float(val) for val in np.linspace(0, 1, 11)]
        TestSHRINK().run_shrink_test(
            [test_file]*len(base_epsilons), residual_epsilons, base_epsilons, True
        )


    """
    Part 5: Inferring model on SHRINK compressed then uncompressed test data.
    """
    results: list[tuple[str, float, float]] = []  # List to store (filename, rmse) tuples
    compressed_test_files = [f for f in os.listdir("data/decompressed") if DATASET in f and DIMENSION in f]
    for file in compressed_test_files:
        X_test_comp, y_test_comp = generate_X_y("data/decompressed/" + file, WINDOW_LENGTH, HORIZON, STRIDE) # type: ignore
        try:
            inference_start = time.perf_counter()
            probas, _, preds = learn.get_X_preds(X_test_comp, bs=BATCH_SIZE)
            compressed_inference_time = time.perf_counter() - inference_start
            rmse_test_comp = skm.root_mean_squared_error(y_test_comp, preds) # type: ignore
            results.append((file, rmse_test_comp, compressed_inference_time))  # Add filename and RMSE to results
        except RuntimeError as e:
            print(e)
            print(f"Too few data points in time series to generate one batch for file: {file}.")

    """
    Part 6: Save statistics in CSV.
    """
    with open(f"data/tsai_results/{DATASET}_dim{DIMENSION}_{str(CHOSEN_MODEL.__name__)}_wl{WINDOW_LENGTH}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'rmse', 'duration'])  # Header
        writer.writerow([train_file, rmse_val, training_time]) # Train rmse
        writer.writerow([test_file, rmse_test, uncompressed_inference_time]) # Test uncompressed rmse
        writer.writerows(results)  # Write all rows
