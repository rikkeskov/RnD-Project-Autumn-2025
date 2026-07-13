"""
Test module for Shrink Python implementation.
"""

import csv
import os
import sys
import unittest

import numpy as np
import pandas as pd # type: ignore
from typing import List

# Ensure project root is on sys.path so absolute imports work when running this module as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quan_trc import compress
from shrink.constants import (
    BASE_FOLDER,
    TURBO_RANGE_CODER_CODES_BASE_PATH
)
from shrink.shrink import Shrink
from shrink.time_series_reader import TimeSeriesReader
from shrink.time_series import TimeSeries


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
    ) -> None:
        """
        The entrance function to extact base and residuals for datasets
        Parameters:
            - filenames: list of the files
            - epsilons: list of the desired epsilon for compression
        """
        if save:
            with open("data/shrink_results/"+filenames[0].split("/")[1].split(".")[0]+"_results.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                writer.writerow(["filename",
                                    "epsilon_pct",
                                    "base_epsilon",
                                    "compression_ratio",
                                    "compression_time",
                                    "decompression_time"
                                    ])
        ts: TimeSeries = TimeSeries(data=[], data_range=0.0)

        for i, filename in enumerate(filenames):
            print(f"File: {filename}")
            # 0. Set Base error
            base_epsilon: float = base_epsilons[i]
            print(f"Shrink: BaseEpsilon = {base_epsilon}")

            # 1. Read dataset
            ts = TimeSeriesReader.get_time_series(filename)
            print(f"{filename}: {ts.size/1024/1024:.2f}MB")

            # 2. Extract Base
            shrink: Shrink = Shrink(points=ts.data, epsilon=base_epsilon)

            binary = shrink.to_byte_array(variable_byte=False, zstd=False)
            original_base_size = shrink.save_bytes(binary, filename)

            # 3. Entropy coding for Base
            inpath = BASE_FOLDER + "/" + filename.split(".")[0].split("/")[1] + "_base.bin"
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
                    with open("data/shrink_results/"+filenames[0].split("/")[1].split(".")[0]+"_results.csv", "a", newline="") as csvfile:
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
                assert self.ts_decompressed is not None
                timestamps = [point.timestamp for point in self.ts_decompressed]
                values = [point.value for point in self.ts_decompressed]
                with open("data/decompressed/"+filename.split("/")[1].split(".")[0]+"_e"+str(epsilon_pct)+"_eb"+str(base_epsilon)+"_decompressed.csv", "w", newline="") as csvfile:
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
        return


class BaseEpsilonCalculation():
    def __init__(self, file_path: str) -> None:
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
        
        self.v = np.asarray(values, dtype=float)


    def _snr(self, tau: int) -> float: # type: ignore
        """Signal-to-Noise Ratio (Eq. 2) for a given quantization level tau."""
        numerator = np.sum(self.v ** 2) # type: ignore
        quantized = np.floor(self.v * 2.0 ** (-tau)) * 2.0 ** tau
        denominator = np.sum((self.v - quantized) ** 2)

        if denominator == 0:
            return np.inf  # perfect reconstruction at this tau
        return 10 * np.log10(numerator / denominator) # type: ignore


    def _initial_tau(self, eta: float) -> int: # type: ignore
        """Initial quantization level tau (Eq. 3)."""
        n = len(self.v) # type: ignore
        sum_v2 = np.sum(self.v ** 2) # type: ignore
        tau0 = np.floor(0.5 * np.log2((10 ** (-eta / 10)) * sum_v2 / n)) + 1 # type: ignore
        return int(tau0)

    def compute_epsilon_base(self, eta: float) -> float:
        """
        Compute the default (Base) quantization error threshold epsilon_b.

        This follows the paper's SNR-driven search:
        1. Get an initial tau from Eq. 3.
        2. Increase tau step by step, recomputing SNR (Eq. 2) each time,
            until the SNR drops below the target eta.
        3. The last tau for which SNR >= eta is the chosen quantization level.
        4. epsilon_b = 2^tau (from comparing Eq. 1 and Eq. 4).

        Parameters
        ----------
        v : array-like
            The data series (e.g. one shrinking-cone interval / the whole series).
        eta : float
            Target SNR threshold (dB) that the quantization must maintain.

        Returns
        -------
        float
            epsilon_b, the default quantization error threshold.
        """
    
        tau = self._initial_tau(eta)

        # Search upward while SNR still meets/exceeds the target.
        while self._snr(tau) >= eta:
            tau += 1

        # Step back to the last tau that satisfied SNR >= eta.
        tau -= 1

        epsilon_b = 2.0 ** tau
        print(f"e_b: {epsilon_b}")
        return epsilon_b

