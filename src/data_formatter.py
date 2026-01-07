import os
import pandas as pd
import sys
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

def find_missing_values():
    folder_path = "./data"

    files = os.listdir(folder_path)

    # Iterate over each file
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            try:
                # Load the CSV file into a DataFrame
                df = pd.read_csv(file_path) # type: ignore
                print(f"\nFile: {file}")

                # Check for null entries
                null_counts = df.isnull().sum()
                if null_counts.sum() > 0:
                    print("Null entries found:")
                    print(null_counts[null_counts > 0])
                else:
                    print("No null entries found.")
            except Exception as e:
                print(f"Error reading {file}: {e}")