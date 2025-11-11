"""
Test module for Shrink Python implementation.
"""

import os
import unittest
from typing import List

import matplotlib.pyplot as plt

from .quan_trc import compress
from .shrink.constants import (
    BASE_FOLDER,
    DATA_PATH,
    TURBO_RANGE_CODER_CODES_BASE_PATH,
)
from .shrink.shrink import Shrink
from .shrink.time_series_reader import TimeSeriesReader
from .shrink.time_series import TimeSeries
from .shrink.shrink_segment import ShrinkSegment
from .shrink.point import Point


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
        self, filenames: List[str], epsilons: list[float], base_epsilons: list[float]
    ) -> tuple[list[Point], list[list[ShrinkSegment]]]:
        """
        The entrance function to extact base and residuals for datasets
        Parameters:
            - filenames: list of the files
            - epsilons: list of the desired epsilon for compression
        """
        results: list[list[ShrinkSegment]] = []
        ts: TimeSeries = TimeSeries(data=[], data_range=0.0)

        for i, filename in enumerate(filenames):
            # 0. Set Base error
            base_epsilon: float = base_epsilons[i]
            print(f"Shrink: BaseEpsilon = {base_epsilon}")

            # 1. Read dataset
            ts = TimeSeriesReader.get_time_series(DATA_PATH + filename)
            print(f"{filename}: {ts.size/1024/1024:.2f}MB")

            # 2. Extract Base
            shrink: Shrink = Shrink(points=ts.data, epsilon=base_epsilon)

            shrink_segments = shrink.segments
            shrink_segments.sort(
                key=lambda segment: (
                    segment.init_timestamp,
                    segment.get_b,
                    segment.get_a,
                )
            )
            results.append(shrink_segments)
            binary = shrink.to_byte_array(variable_byte=False, zstd=False)
            original_base_size = shrink.save_bytes(binary, filename)

            # 3. Entropy coding for Base
            inpath = BASE_FOLDER + filename[:-7] + "_base.bin"
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

                mean_compression_ratio += compression_ratio
                mean_result_compression_ratio += residual_compression_ratio
                mean_compression_time += residual_time
                mean_decoding_time += self.decompression_results_time

            mean_compression_time, mean_decoding_time = mean_compression_time / len(
                epsilons
            ), (mean_decoding_time + self.decompression_base_time) / len(epsilons)
            mean_compression_ratio, mean_result_compression_ratio = (
                mean_compression_ratio / len(epsilons),
                mean_result_compression_ratio / len(epsilons),
            )
            print(f"The average compresstime: {mean_compression_time:.1f}ms \n")
        return ts.data, results

    def plot_shrink_decompressed(
        self, original_data: list[Point], segment_results: list[list[ShrinkSegment]]
    ) -> None:
        """Plot the decompressed values."""

        original_timestamps = [val.timestamp for val in original_data]
        original_values = [val.value for val in original_data]
        plt.plot(original_timestamps, original_values, marker=".", color="black")  # type: ignore

        # if self.ts_decompressed is None:
        #     return None
        # shrink_timestamps = [val.timestamp for val in self.ts_decompressed]
        # shrink_values = [val.value for val in self.ts_decompressed]
        # plt.figure()
        # plt.plot(shrink_timestamps, shrink_values, marker=".", color="black")

        for segment_list in segment_results:
            timestamps = [val.init_timestamp for val in segment_list]
            values = [val.get_b for val in segment_list]
            plt.plot(  # type: ignore
                timestamps,
                values,
                linestyle="--",
                marker="+",
                label="Segment",
                markersize=12,
            )
        plt.legend(  # type: ignore
            ["Original data", "Base epsilon=0.01", "Base epsilon=0.5"], loc="upper left"
        )
        plt.show()  # type: ignore


if __name__ == "__main__":
    files = [
        "/HouseholdPowerConsumption1Row1.csv",
        "/HouseholdPowerConsumption1Row1.csv",
    ]  # , "MoteStrain.csv", "Lightning.csv", "Cricket.csv"]
    in_base_epsilons = [
        0.1,
        0.2,
    ]  # [0.525],          0.85,             1.235,           1.14]
    in_epsilons = [
        0.01,
        0.0075,
        2.0,
        1.0,
    ]  # [0.01, 0.0075], 0.005, 0.0025, 0.001, 0.00075, 0.0001, 0]
    test = TestSHRINK()
    originaldata, test_results = test.run_shrink_test(
        files, in_epsilons, in_base_epsilons
    )
    test.plot_shrink_decompressed(originaldata, test_results)
