import glob
import os
import re

import pandas as pd
from typing import Any


RESULTS_ROOT = "results"
OUTPUT_FILE = "all_results.csv"


KNOWN_MODELS = [
    "InceptionTime",
    "MLP",
    "FCN"
]

KNOWN_PREPROCESSING = [
    "INTERPOLATION",
    "KNN",
    "MEAN",
]


def extract_parameters(filepath: str) -> dict[str, Any]:

    filename = os.path.basename(filepath)
    folder = os.path.basename(os.path.dirname(filepath))

    # ------------------------
    # folder parameters
    # ------------------------

    wl = re.search(r"window_length(\d+)", folder)
    cycles = re.search(r"cycles(\d+)", folder)
    bs = re.search(r"batchsize(\d+)", folder)

    # ------------------------
    # filename parameters
    # ------------------------

    dataset = None
    dimension = None
    model = None
    preprocessing = None

    m = re.search(r"(.+?)_dim([^_]+)_([^_]+)_([^_]+)", filename)

    if m:
        dataset = m.group(1)
        dimension = m.group(2)
        model = m.group(3)
        preprocessing = m.group(4) if m.group(4) != "combined.csv" else None


    return {
        "dataset": dataset,
        "dimension": dimension,
        "model": model,
        "preprocessing": preprocessing,
        "window_length": int(wl.group(1)) if wl else None,
        "cycles": int(cycles.group(1)) if cycles else None,
        "batch_size": int(bs.group(1)) if bs else None,
    }


def add_calculated_columns(df: pd.DataFrame):

    filename = df["filename"].fillna("")

    df.loc[ # type: ignore
        filename.str.contains("_TRAIN_", case=False),
        "split"
    ] = "train"

    df.loc[ # type: ignore
        filename.str.contains("_TEST_", case=False),
        "split"
    ] = "test"

    # compressed?
    df["compressed"] = filename.str.contains(
        "decompressed",
        case=False,
    )
    return df


def build_master_table(results_root: str) -> pd.DataFrame:

    csvs = glob.glob(
        os.path.join(results_root, "**", "*combined.csv"),
        recursive=True,
    )

    all_results: list[pd.DataFrame] = []

    for file in csvs:
        df = pd.read_csv(file)
        params = extract_parameters(file)

        for key, value in params.items():
            df[key] = value

        df = add_calculated_columns(df)
        all_results.append(df) # type: ignore

    master = pd.concat( # type: ignore
        all_results,
        ignore_index=True,
    )

    assert isinstance(master, pd.DataFrame)

    return master


if __name__ == "__main__":

    master = build_master_table(RESULTS_ROOT)

    master.to_csv(
        OUTPUT_FILE,
        index=False,
    )