"""
Test module for Shrink Python implementation.
"""

import csv
import os
import sys
import unittest

import pandas as pd # type: ignore
from typing import List

# Ensure project root is on sys.path so absolute imports work when running this module as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quan_trc import compress
from shrink.constants import (
    BASE_FOLDER,
    DATA_PATH,
    TURBO_RANGE_CODER_CODES_BASE_PATH
)
from shrink.shrink import Shrink
from shrink.time_series_reader import TimeSeriesReader
from shrink.time_series import TimeSeries
from shrink.shrink_segment import ShrinkSegment
from shrink.point import Point

from data_preprocessing import missing_value_count, preprocessing_and_save_as_csv, PreprocessingType

PREPROCESSING_TYPE = PreprocessingType.INTERPOLATION


class TestSHRINK(unittest.TestCase):
    """
    Unittest of Shrink.
    """

    def __init__(self):
        """
        The initial function of the class
        Parameters:
        """
        super().__init__()
        self.duration = 0
        self.ts_decompressed = None
        self.decompression_base_time = 0
        self.decompression_results_time = 0

    def assert_shrink(self, shrink: Shrink, epsilon_pct: float, ts: TimeSeries):
        """
        Decompress the data into Base and residual.
        Assert the values are within the error threshold
        Parameters:
            - shrink: the algorithm
            - epsilonPct: current epsilon for compression
            - ts: the time series data
        """
        idx = 0
        self.ts_decompressed, self.decompression_base_time = shrink.decompress()
        dequantized_value, self.decompression_results_time = shrink.residual_decode(
            epsilon=epsilon_pct
        )

        for expected in self.ts_decompressed:
            actual = ts.data[idx]
            approximate_value = expected.value + dequantized_value[idx]
            if expected.timestamp != actual.timestamp:
                continue
            if epsilon_pct == 0:
                # You can also use 1e-10, which is considered equal to sys.float_info.epsilon
                self.assertAlmostEqual(
                    actual.value,
                    approximate_value,
                    delta=1e-10,
                    msg="Value did not match for timestamp " + str(actual.timestamp),
                )
            else:
                self.assertAlmostEqual(
                    actual.value,
                    approximate_value,
                    delta=epsilon_pct,
                    msg="Value did not match for timestamp " + str(actual.timestamp),
                )
            idx += 1
        self.assertEqual(idx, len(ts.data))

    def run_shrink_test(
        self, filenames: List[str], epsilons: list[float], base_epsilons: list[float], save: bool
    ) -> tuple[list[Point], list[list[ShrinkSegment]]]:
        """
        The entrance function to extact base and residuals for datasets
        Parameters:
            - filenames: list of the files
            - epsilons: list of the desired epsilon for compression
        """
        if save:
            with open("data/shrink_results/"+filenames[0].split("/")[2].split(".")[0]+"_results.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                writer.writerow(["filename",
                                    "epsilon_pct",
                                    "base_epsilon",
                                    "compression_ratio",
                                    "compression_time",
                                    "decompression_time"
                                    ])
        results: list[list[ShrinkSegment]] = []
        ts: TimeSeries = TimeSeries(data=[], data_range=0.0)

        for i, filename in enumerate(filenames):
            print(f"File: {filename}")
            # 0. Set Base error
            base_epsilon: float = base_epsilons[i]
            print(f"Shrink: BaseEpsilon = {base_epsilon}")

            # 1. Read dataset
            ts = TimeSeriesReader.get_time_series(DATA_PATH + filename)
            print(f"{filename}: {ts.size/1024/1024:.2f}MB")

            # 2. Extract Base
            shrink: Shrink = Shrink(points=ts.data, epsilon=base_epsilon)

            shrink_segments = shrink.segments
            # shrink_segments.sort(
            #     key=lambda segment: (
            #         segment.init_timestamp,
            #         segment.get_b,
            #         segment.get_a,
            #     )
            # )
            results.append(shrink_segments)
            binary = shrink.to_byte_array(variable_byte=False, zstd=False)
            original_base_size = shrink.save_bytes(binary, filename)

            # 3. Entropy coding for Base
            inpath = BASE_FOLDER + "/" + filename.split(".")[0].split("/")[2] + "_base.bin"
            compress(inpath, TURBO_RANGE_CODER_CODES_BASE_PATH)
            base_time = int(shrink.base_time)
            base_size = os.path.getsize(TURBO_RANGE_CODER_CODES_BASE_PATH)

            # 4. Get Residuals
            residuals = shrink.get_residuals()

            # 5. Encoding for different epsilons
            mean_compression_ratio, mean_result_compression_ratio = 0, 0
            mean_compression_time, mean_decoding_time, decoding_base_time = (
                base_time,
                0,
                0,
            )
            decoding_base = False
            for epsilon_pct in epsilons:
                if epsilon_pct >= base_epsilon:
                    print(
                        f"Epsilon: {epsilon_pct }\t "
                        + f"Compression Ratio: {ts.size/base_size :.5f}\t"
                        + f"Residual CR: {0}\t"
                        + f"Compress Time: {base_time}ms\t "
                        + f"Decompress Time: {decoding_base_time} "
                        + f"+ {self.decompression_results_time} = "
                        + f"{self.decompression_base_time +self.decompression_results_time}ms\t"
                        + f"Range: {ts.data_range :.3f}"
                    )
                    print(
                        f"baseSize: {base_size/1024 :.3f}KB \t "
                        + f"Size of residual: {0}KB \t origibaseSize: {original_base_size/1024}KB"
                    )
                    mean_compression_ratio += ts.size / base_size
                    mean_result_compression_ratio += 0
                    continue

                residual_size = shrink.residual_encode(residuals, epsilon_pct)
                residual_time = shrink.residual_time

                compressed_size = base_size + residual_size
                residual_compression_ratio = ts.size / residual_size
                compression_ratio = ts.size / compressed_size

                if decoding_base is False:
                    # To decompress the Base only one,
                    # we should assert error is bounded with current errorthreshold epsilonPct
                    self.assert_shrink(shrink, epsilon_pct, ts)
                    decoding_base = True
                    decoding_base_time = self.decompression_base_time

                print(
                    f"Epsilon: {epsilon_pct }\t"
                    + f"Compression Ratio: {compression_ratio:.5f} \t "
                    + f"baseSize: {base_size/1024 :.3f}KB \t "
                    + f"residualSize: {residual_size/1024 :.3f}KB \t"
                    + f"Compress Time: {base_time} + {residual_time} = "
                    + f"{base_time + residual_time}ms\t "
                    + f"Decompress Time: {decoding_base_time} + "
                    + f"{self.decompression_results_time} = "
                    + f"{self.decompression_base_time +self.decompression_results_time}ms"
                )
                if save:
                    with open("data/shrink_results/"+filenames[0].split("/")[2].split(".")[0]+"_results.csv", "a", newline="") as csvfile:
                        writer = csv.writer(csvfile, delimiter=",")
                        writer.writerow([filename,
                                        epsilon_pct,
                                        base_epsilon,
                                        compression_ratio,
                                        base_time + residual_time,
                                        self.decompression_base_time +self.decompression_results_time
                                        ])

                mean_compression_ratio += compression_ratio
                mean_result_compression_ratio += residual_compression_ratio
                mean_compression_time += residual_time
                mean_decoding_time += self.decompression_results_time

                # Save decompressed data
                for segment_list in results:
                    timestamps = [val.init_timestamp for val in segment_list]
                    values = [val.get_b for val in segment_list]
                    with open("data/decompressed/"+filename.split("/")[2].split(".")[0]+"_e"+str(epsilon_pct)+"_eb"+str(base_epsilon)+"_decompressed.csv", "w", newline="") as csvfile:
                        writer = csv.writer(csvfile, delimiter=",")
                        for t, v in zip(timestamps, values):
                            writer.writerow([t, v])

            mean_compression_time, mean_decoding_time = mean_compression_time / len(
                epsilons
            ), (mean_decoding_time + self.decompression_base_time) / len(epsilons)
            mean_compression_ratio, mean_result_compression_ratio = (
                mean_compression_ratio / len(epsilons),
                mean_result_compression_ratio / len(epsilons),
            )
            print(f"The average compresstime: {mean_compression_time:.1f}ms \n")
        return ts.data, results

def calc_epsilon_base(file_path: str, percentages: list[float]) -> list[float]:
    """
    Calculate the base error threshold (epsilon_b) as a fraction of the data range from a CSV file.

    Parameters:
    - file_path: Path to the CSV file containing the data.
    - percentage: The fraction of the data range to use for epsilon_b (e.g., 5 for 5%).

    Returns:
    - epsilon_b: The calculated base error threshold.
    """
    values: list[float] = []
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                try:
                    value = float(row[1])
                except ValueError as e:
                    continue
                values.append(value)
    except OSError as e:
        print(e)
        raise OSError("See print.") from e
    
    # Calculate the data range
    data_range = max(values) - min(values)

    # Calculate epsilon_b
    epsilon_b: list[float] = []
    for pct in percentages:
        epsilon_b.append(round(pct * data_range, 4))
    return epsilon_b

def count_decimal_places(file_path: str) -> int:
    # Load the CSV file
    df = pd.read_csv(file_path, header=None, names=['timestamp', 'value']) # type: ignore

    # Convert 'value' column to string to count decimal places
    df['value'] = df['value'].astype(str) # type: ignore

    # Function to count decimal places
    def decimal_places(value: str):
        if '.' in value:
            return len(value.split('.')[1].rstrip('0'))
        else:
            return 0

    # Apply the function to each value
    df['decimal_places'] = df['value'].apply(decimal_places) # type: ignore

    # Count the maximum number of decimal places
    max_decimal_places = df['decimal_places'].max()

    return max_decimal_places

def process_directory(directory_path: str):
    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            max_decimal_places = count_decimal_places(file_path)
            print(f"File: {filename}, Maximum number of decimal places: {max_decimal_places}")


if __name__ == "__main__":
    files = [
        "/HouseholdPowerConsumption1_TEST_dim1.csv",
    ]
    base_percentages = [0.01, 0.02, 0.05, 0.075, 0.1, 0.15]
    num_files = len(base_percentages)

    for filename in files:
        train_df: pd.DataFrame = pd.read_csv("data" + filename, sep=",", header=None, index_col=0)
        nan_count: int = missing_value_count(train_df)
        if nan_count > 0:
            filename = preprocessing_and_save_as_csv(filename, PREPROCESSING_TYPE)
            train_df: pd.DataFrame = pd.read_csv(filename, sep=",", header=None, index_col=0)
        nan_count: int = missing_value_count(train_df)
        if nan_count > 0:
            print("Error in preprocessing.")
            exit()
        files = [filename[4:]]   * num_files
        in_base_epsilons = calc_epsilon_base(filename, base_percentages)

        num_decimals = count_decimal_places(filename)
        print(f"Number of decimals for file: {filename} is {num_decimals}.")
        if num_decimals < 3:
            in_epsilons = [0.01, 0.0075, 0.005, 0.0025, 0.001]
        else:
            in_epsilons = [0.01, 0.0075, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.00025, 0.0001] # when decimal >= 3
        print(f"Epsilons are therefore {in_epsilons}.")
        test = TestSHRINK()
        originaldata, test_results = test.run_shrink_test(
            files, in_epsilons, in_base_epsilons, True
        )

