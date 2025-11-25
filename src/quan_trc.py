"""Module for using turbo range coder."""

from typing import Any
from timeit import default_timer as timer
import os
import sys
import csv
import time
import numpy as np

# Ensure project root is on sys.path so absolute imports work when running this module as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shrink.constants import (
    QUANTRC_NORMALIZATION,
    TURBO_RANGE_CODER_CODES_OUT_PATH,
    TURBO_CODE_LOCATION,
    TURBO_CODE_PARAMETER,
    DATA_PATH,
    TURBO_RANGE_CODER_NEWORIGINAL_OUT_PATH,
)


def load(data: np.ndarray[Any, Any]) -> dict[str, float]:
    """
    Loads a dataset in for compression, data is assumed
    to be a numpy array. Normalizes the data between 0 and
    1 automatically.
    """
    compression_stats: dict[str, float] = {}
    start = timer()

    # store the variables
    data = data.copy()
    n, p = data.shape

    normalization = np.vstack([np.max(data, axis=0), np.min(data, axis=0)])
    normalization = np.hstack([normalization, np.array([n, p]).reshape(2, 1)])

    # get all attrs between 0 and 1
    for i in range(p):
        data[:, i] = (data[:, i] - normalization[1, i]) / (
            normalization[0, i] - normalization[1, i]
        )

    np.save(QUANTRC_NORMALIZATION, normalization)
    compression_stats["load_time"] = timer() - start
    compression_stats["original_size"] = data.size * data.itemsize
    return compression_stats


def compress(in_path: str, out_path: str) -> None:
    """Command line compression."""
    command = f"{TURBO_CODE_LOCATION} -{TURBO_CODE_PARAMETER} {in_path} {out_path}"
    print("command: ", command)
    os.system(command)


def decompress(compressed_path: str, out_path: str) -> None:
    """Command line decompression."""
    command = f"{TURBO_CODE_LOCATION} -d {compressed_path} {out_path}"
    print("command: ", command)
    os.system(command)


def calculate_compression_ratio(
    original_file_path: str, compressed_file_path: str
) -> float:
    "Calculate compression ratio."
    # print("Original size: ", os.path.getsize(originalFile))
    # print("CompressedSize", os.path.getsize(compressedFile))
    return os.path.getsize(compressed_file_path) / os.path.getsize(original_file_path)


def equal_or_not(file1_path: str, file2_path: str) -> None:
    "Compare files."
    # Create two empty lists to store the data in the file
    data1: list[float] = []
    data2: list[float] = []

    # Open the first CSV file, read the data and store it in the data1 list
    with open(file1_path, mode="r", newline="", encoding="utf-8") as file1:
        csv_reader = csv.reader(file1)
        for row in csv_reader:
            data1.append(float(row[0]))

    # Open the second CSV file, read the data and store it in the data2 list
    with open(file2_path, mode="r", newline="", encoding="utf-8") as file2:
        csv_reader = csv.reader(file2)
        for row in csv_reader:
            data2.append(float(row[0]))

    # Compare two lists to see if they are the same
    if data1 == data2:
        print("The two files are Equal")
    else:
        print("The two files are Not Equal")

        # Find and indicate the row numbers and values of the first 10 unequal rows
        num_differences = 0
        for i in range(min(len(data1), len(data2))):
            if data1[i] != data2[i]:
                num_differences += 1
                print(
                    f"Line number {i + 1}: value is {data1[i]} (file 1) and {data2[i]} (file 2)"
                )
                if num_differences >= 10:
                    break


if __name__ == "__main__":
    filenames = ["/FaceFour.csv"]

    for filename in filenames:
        print(filename)
        start_time = time.time()
        compress(DATA_PATH + filename, TURBO_RANGE_CODER_CODES_OUT_PATH)
        end_time = time.time()
        duration = int((end_time - start_time) * 1000)

        sourceSize = os.path.getsize(DATA_PATH + filename)
        compressSize = os.path.getsize(TURBO_RANGE_CODER_CODES_OUT_PATH)
        print(
            f"Original size: {sourceSize/1024}KB  CompressedSize: {compressSize/1024}KB"
        )
        print(
            "Compression ratio: ",
            calculate_compression_ratio(
                DATA_PATH + filename, TURBO_RANGE_CODER_CODES_OUT_PATH
            ),
            f"  Execution Time: {duration}ms ",
        )

        start_time = time.time()
        decompress(
            TURBO_RANGE_CODER_CODES_OUT_PATH,
            TURBO_RANGE_CODER_NEWORIGINAL_OUT_PATH,
        )
        end_time = time.time()
        decompResTime = int((end_time - start_time) * 1000)
        print(f"Time of decompressing: {decompResTime:.2f}")
