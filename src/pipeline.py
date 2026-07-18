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
DATASET = "HouseholdPowerConsumption1"
DIMENSIONS = ["0", "1", "4"]

# X y generation settings
WINDOW_LENGTH = 150
HORIZON = 1
STRIDE = None

# ML model settings
CHOSEN_MODEL = InceptionTime
BATCH_SIZE = 16
CYCLES = 50

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

def save_combined_results(
    regression_results: list[tuple[str, float, float]],
    train_file: str,
    test_file: str,
    train_rmse: float,
    test_rmse: float,
    training_time: float,
    test_inference_time: float,
    output_file: str
):
    """
    Combine SHRINK statistics and regression results into one CSV.

    SHRINK csv:
    filename, base_epsilon, compression_ratio,
    compression_time, decompression_time

    Regression:
    filename, rmse, inference_time
    """

    # Load SHRINK results
    try:
        shrink_df = pd.read_csv(SHRINK_RESULTS_FILE_1)
    except FileNotFoundError:
        print("Finding prepocessed test data SHRINK results.")
        shrink_df = pd.read_csv(SHRINK_RESULTS_FILE_2)
    except:
        print("could not open shrink results.")
        exit()

    combined_results = []

    # Add original training result
    combined_results.append({ # type: ignore
        "filename": train_file,
        "base_epsilon": None,
        "compression_ratio_base_only": None,
        "compression_ratio": None,
        "compression_time": None,
        "decompression_time": None,
        "rmse": train_rmse,
        "inference_time": training_time
    })

    # Add original test result
    combined_results.append({ # type: ignore
        "filename": test_file,
        "base_epsilon": 0,
        "compression_ratio_base_only": 1,
        "compression_ratio": 1,
        "compression_time": 0,
        "decompression_time": 0,
        "rmse": test_rmse,
        "inference_time": test_inference_time
    })


    # Add SHRINK results
    for filename, rmse, inference_time in regression_results:

        # Example filename:
        # BIDMC32_TEST_dimAVR_e0.0_eb0.0625_decompressed.csv

        base_epsilon = float(
            filename.split("_eb")[1]
            .split("_decompressed")[0]
        )

        shrink_row = shrink_df[ # type: ignore
            (shrink_df["base_epsilon"] == base_epsilon)
        ]

        if shrink_row.empty: # type: ignore
            print(
                f"Warning: no SHRINK result found for {filename}"
            )
            continue

        shrink_row = shrink_row.iloc[0] # type: ignore

        combined_results.append({ # type: ignore
            "filename": filename,
            "base_epsilon": shrink_row["base_epsilon"],
            "compression_ratio_base_only": shrink_row["compression_ratio_base_only"],
            "compression_ratio": shrink_row["compression_ratio"],
            "compression_time": shrink_row["compression_time"],
            "decompression_time": shrink_row["decompression_time"],
            "rmse": rmse,
            "inference_time": inference_time
        })


    combined_df = pd.DataFrame(combined_results) # type: ignore

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    combined_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved combined results to {output_file}"
    )

if __name__ == "__main__":
    for dimension in DIMENSIONS:
        test_file = "data/" + DATASET + "_TEST_dim" + dimension + ".csv"
        train_file = "data/" + DATASET + "_TRAIN_dim" + dimension + ".csv"
        SHRINK_RESULTS_FILE_1 = f"data/shrink_results/{DATASET}_TEST_dim{dimension}_results.csv"
        SHRINK_RESULTS_FILE_2 = f"data/shrink_results/{DATASET}_TEST_dim{dimension}_{PREPROCESSING_TYPE.name}_results.csv"
        """
        Part 1: Preprocessing. Filling missing values in training set
        """
        train_file = preprocessing(train_file, PREPROCESSING_TYPE)
        X_train, y_train = generate_X_y(train_file, WINDOW_LENGTH, HORIZON, STRIDE) # type: ignore
        splits_train = make_ts_splits(X_train, valid_pct=0.2)
        print("Training samples:", len(splits_train[0]))
        print("Validation samples:", len(splits_train[1]))

        """
        Part 2: Training model on training data. Generate X and y, generate validation set
        """

        tfms  = [None, [TSRegression()]] # type: ignore
        batch_tfms = TSStandardize(by_sample=True, by_var=True)
        dls: TSDataLoaders = get_ts_dls(
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
        path_model = Path(f"./{DATASET}_dim{dimension}_rmse{round(rmse_val, 5)}_{str(CHOSEN_MODEL.__name__)}.pkl")
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
        test_file = preprocessing(test_file, PREPROCESSING_TYPE)
        X_test, y_test = generate_X_y(test_file, WINDOW_LENGTH, HORIZON, STRIDE) # type: ignore
        print(f"Shape of X_test: {X_test.shape}, type: {type(X_test)} and length: {len(X_test)}.")
        print(f"Number of variables in learner: {learn.dls.vars} and length of data loader: {learn.dls.len}.")
        inference_start = time.perf_counter()
        probas, _, preds = learn.get_X_preds(X_test, bs=BATCH_SIZE)
        uncompressed_inference_time = time.perf_counter() - inference_start
        rmse_test = skm.root_mean_squared_error(y_test, preds) # type: ignore

        """
        Part 4: Compress and decompress test data.
        """
        scaling_factors = [20, 25, 30, 35, 40]
        calculator = BaseEpsilonCalculation(test_file)
        base_epsilons = [
            calculator.compute_epsilon_base(f)
            for f in scaling_factors
        ]
        TestSHRINK().run_shrink_test(
            [test_file]*len(base_epsilons), [0], base_epsilons, True
        )


        """
        Part 5: Inferring model on SHRINK compressed then uncompressed test data.
        """
        results: list[tuple[str, float, float]] = []  # List to store (filename, rmse) tuples
        compressed_test_files = [
            f for f in os.listdir("data/decompressed")
            if DATASET in f and f"_dim{dimension}_" in f
        ]
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
        Part 6: Save combined statistics.
        """

        save_combined_results(
            regression_results=results,
            train_file=train_file,
            test_file=test_file,
            train_rmse=rmse_val,
            test_rmse=rmse_test,
            training_time=training_time,
            test_inference_time=uncompressed_inference_time,
            output_file=f"data/results/{DATASET}_dim{dimension}_{CHOSEN_MODEL.__name__}_combined.csv"
        )
