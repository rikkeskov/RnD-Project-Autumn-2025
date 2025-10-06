"""
Shrink, Data Compression by Semantic Extraction and Residuals Encoding.

Python implementation. See experiments for use.
"""

import csv
import math
import os
import time

from io import BytesIO
from typing import List

from quan_trc import compress, decompress

from .constants import (
    BASE_FOLDER,
    TURBO_RANGE_CODER_CODES_RESIDUALS_PATH,
    OUTPUT_PATH_RESIDUALS,
    OUTPUT_PATH_BASE,
    OUTPUT_PATH_DEQUANTIRESIDUALS,
    OUTPUT_PATH_NEWORIGINAL,
    OUTPUT_PATH_QUANTIRESIDUALS,
)
from .float_encoder import FloatEncoder
from .point import Point
from .shrink_segment import ShrinkSegment
from .uint_encoder import UIntEncoder
from .utility_functions import (
    res_dequantize,
    high_precision_subtract,
    high_precision_add,
)
from .variable_byte_encoder import VariableByteEncoder


class Shrink:
    """
    Shrink on args.
    Args:
        points, the timestamp, data,
        epsilon, the allowed error,
        data_bytes, the data if not in points format,
        variable_byte, a boolean value for ??,
        zstd, a boolean value for ???.
    """

    def __init__(
        self,
        points: list[Point] | None = None,
        epsilon: float | None = None,
        data_bytes: bytes | None = None,
        variable_byte: bool = False,
        zstd: bool = False,
    ):
        """
        Initialization, including compression and decompression

        Args:
            points: List[Point]
            epsilon:  ts.range * epsilonPct(0.05)
            data_bytes: bytes=binary

        Returns:

        """
        if points is not None:  # Handle the case where points is a list of Points
            start_time = time.time()

            if not points:
                raise ValueError("No points provided")
            self.alpha = 1  ### hyperparameter for setting L
            self.epsilon = epsilon
            self.last_timestamp = points[-1].timestamp
            self.values = [point.value for point in points]
            self.max, self.min = max(self.values), min(self.values)
            self.length = len(points)
            self.segments_length = None
            self.segments = self.merge_per_b(self.compress(points))
            self.points = points[:]
            self.residual_time = 0
            end_time = time.time()
            self.base_time = int((end_time - start_time) * 1000)
        elif data_bytes is not None:  # Handle the case where bytes is a byte array
            self.read_byte_array(data_bytes, variable_byte, zstd)
        else:
            raise ValueError("Either points or bytes must be provided")

    def get_residuals(self) -> list[float]:
        "Get residuals."
        self.segments.sort(key=lambda segment: segment.get_init_timestamp)
        residuals: list[float] = []
        expected_values: list[float] = []
        expected_points: list[Point] = []
        idx = 0
        current_timestamp = self.segments[0].get_init_timestamp

        for i in range(len(self.segments) - 1):
            while current_timestamp < self.segments[i + 1].get_init_timestamp:
                expected_value = high_precision_add(
                    self.segments[i].get_a
                    * (current_timestamp - self.segments[i].get_init_timestamp),
                    self.segments[i].get_b,
                )
                expected_values.append(expected_value)
                residual_value = high_precision_subtract(
                    self.values[idx], expected_value
                )
                residuals.append(residual_value)
                expected_points.append(Point(current_timestamp, expected_value))
                current_timestamp += 1
                idx += 1

        while current_timestamp <= self.last_timestamp:
            expected_value = high_precision_add(
                self.segments[-1].get_a
                * (current_timestamp - self.segments[-1].get_init_timestamp),
                self.segments[-1].get_b,
            )
            expected_values.append(expected_value)
            residuals.append(high_precision_subtract(self.values[idx], expected_value))
            expected_points.append(Point(current_timestamp, expected_value))
            current_timestamp += 1
            idx += 1

        csv_file_path = OUTPUT_PATH_RESIDUALS
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            for item in residuals:
                csv_writer.writerow([item])
        return residuals

    def residual_encode(self, residuals: list[float], epsilon: float) -> int:
        "Encode residuals."
        start_time = time.time()
        if epsilon != 0:
            quantized_residaul_values = [round((v / epsilon)) for v in residuals]
        else:
            quantized_residaul_values = residuals[:]
        end_time = time.time()
        residual_time = int((end_time - start_time) * 1000)

        with open(
            OUTPUT_PATH_QUANTIRESIDUALS, mode="w", newline="", encoding="utf-8"
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            for item in quantized_residaul_values:
                csv_writer.writerow([item])

        start_time = time.time()
        compress(OUTPUT_PATH_QUANTIRESIDUALS, TURBO_RANGE_CODER_CODES_RESIDUALS_PATH)
        end_time = time.time()
        residual_time += int((end_time - start_time) * 1000)
        self.residual_time = residual_time
        residual_size = os.path.getsize(TURBO_RANGE_CODER_CODES_RESIDUALS_PATH)

        return residual_size

    def residual_decode(self, epsilon: float) -> tuple[list[float], int]:
        "Decode residuals."
        start_time = time.time()
        decompress(TURBO_RANGE_CODER_CODES_RESIDUALS_PATH, OUTPUT_PATH_NEWORIGINAL)
        dequantized_values: list[float] = []
        with open(
            OUTPUT_PATH_NEWORIGINAL, mode="r", newline="", encoding="utf-8"
        ) as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                dequantized_values.append(float(row[0]))

        if epsilon != 0:
            dequantized_values = res_dequantize(dequantized_values, epsilon)

        end_time = time.time()
        decompress_results_time = int((end_time - start_time) * 1000)

        csv_file_path = OUTPUT_PATH_DEQUANTIRESIDUALS
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            for item in dequantized_values:
                csv_writer.writerow([item])

        return dequantized_values, decompress_results_time

    def __dynamic_epsilon(
        self, start_idx: int, points: list[Point]
    ) -> tuple[float, int]:
        "Returns local epsilon and new buffer pointer."
        if self.epsilon is None:
            raise ValueError("Epsilon must be set!")
        buflength = int(self.length * self.alpha * self.epsilon)
        if buflength >= len(points):
            return self.epsilon, len(points) - 1
        buf: list[float] = []
        for i in range(start_idx, start_idx + buflength):
            if i >= len(points):
                break
            buf.append(points[i].value)

        local_max, local_min = max(buf), min(buf)
        c = round(
            math.exp((2 / 3 - (local_max - local_min) / (self.max - self.min))), 3
        )
        localepsilon = round(self.epsilon * c, 3)

        return localepsilon, start_idx + buflength

    def __quantization(self, value: float, local_epsilon: float) -> float:
        "Quantize the value."
        res = round(value / local_epsilon) * local_epsilon
        return res

    def __create_shrink_segment(
        self,
        start_idx: int,
        points: list[Point],
        segments: list[ShrinkSegment],
        local_epsilon: float,
    ) -> tuple[int, ShrinkSegment]:
        "Create the shrink segments."
        init_timestamp = points[start_idx].timestamp
        b = round(self.__quantization(points[start_idx].value, local_epsilon), 1)

        if start_idx + 1 == len(points):  # Case1: only 1 ponit
            segments.append(ShrinkSegment(init_timestamp, -math.inf, math.inf, b))
            segment = ShrinkSegment(init_timestamp, -math.inf, math.inf, b)
            return start_idx + 1, segment

        a_max = ((points[start_idx + 1].value + local_epsilon) - b) / (
            points[start_idx + 1].timestamp - init_timestamp
        )
        a_min = ((points[start_idx + 1].value - local_epsilon) - b) / (
            points[start_idx + 1].timestamp - init_timestamp
        )
        if start_idx + 2 == len(points):  # Case2: only 2 ponits
            segments.append(ShrinkSegment(init_timestamp, a_min, a_max, b))
            segment = ShrinkSegment(init_timestamp, a_min, a_max, b)
            return start_idx + 2, segment

        for idx in range(start_idx + 2, len(points)):  # Case3: more than 2 ponits
            up_value = points[idx].value + local_epsilon
            down_value = points[idx].value - local_epsilon

            up_lim = a_max * (points[idx].timestamp - init_timestamp) + b
            down_lim = a_min * (points[idx].timestamp - init_timestamp) + b

            if down_value > up_lim or up_value < down_lim:
                segments.append(ShrinkSegment(init_timestamp, a_min, a_max, b))
                segment = ShrinkSegment(init_timestamp, a_min, a_max, b)
                return idx, segment

            if up_value < up_lim:
                a_max = max(
                    (up_value - b) / (points[idx].timestamp - init_timestamp), a_min
                )
            if down_value > down_lim:
                a_min = min(
                    (down_value - b) / (points[idx].timestamp - init_timestamp), a_max
                )

        segment = ShrinkSegment(init_timestamp, a_min, a_max, b)
        segments.append(segment)
        return len(points), segment

    def compress(self, points: list[Point]) -> list[ShrinkSegment]:
        "Compress points into shrink segment."
        segments: list[ShrinkSegment] = []
        current_idx = 0
        new_idx = -1
        if self.epsilon is None:
            raise ValueError("Epsilon must be set!")
        local_epsilon = self.epsilon

        while current_idx < len(points):
            if current_idx > new_idx:
                local_epsilon, new_idx = self.__dynamic_epsilon(current_idx, points)
            current_idx, _ = self.__create_shrink_segment(
                current_idx, points, segments, local_epsilon
            )
        return segments

    def merge_per_b(self, segments: list[ShrinkSegment]) -> list[ShrinkSegment]:
        "Merge segments per b."
        a_min_temp = float("-inf")
        a_max_temp = float("inf")
        b = float("nan")
        timestamps: list[int] = []
        merged_segments: list[ShrinkSegment] = []
        self.segments_length = 0

        segments.sort(key=lambda segment: (segment.get_b, segment.get_a))

        for i, segment in enumerate(segments):
            if b != segment.get_b:
                if len(timestamps) == 1:
                    merged_segments.append(
                        ShrinkSegment(timestamps[0], a_min_temp, a_max_temp, b)
                    )
                    self.segments_length += 1
                else:
                    for timestamp in timestamps:
                        merged_segments.append(
                            ShrinkSegment(timestamp, a_min_temp, a_max_temp, b)
                        )
                        self.segments_length += 1
                timestamps.clear()
                timestamps.append(segment.get_init_timestamp)
                a_min_temp = segment.get_a_min
                a_max_temp = segment.get_a_max
                b = segment.get_b
                continue

            if segment.get_a_min <= a_max_temp and segment.get_a_max >= a_min_temp:
                timestamps.append(segment.get_init_timestamp)
                a_min_temp = max(a_min_temp, segment.get_a_min)
                a_max_temp = min(a_max_temp, segment.get_a_max)
            else:
                if len(timestamps) == 1:
                    merged_segments.append(segments[i - 1])
                    self.segments_length += 1
                else:
                    for timestamp in timestamps:
                        merged_segments.append(
                            ShrinkSegment(timestamp, a_min_temp, a_max_temp, b)
                        )
                        self.segments_length += 1

                timestamps.clear()
                timestamps.append(segment.get_init_timestamp)
                a_min_temp = segment.get_a_min
                a_max_temp = segment.get_a_max

        if timestamps:
            if len(timestamps) == 1:
                merged_segments.append(
                    ShrinkSegment(timestamps[0], a_min_temp, a_max_temp, b)
                )
                self.segments_length += 1
            else:
                for timestamp in timestamps:
                    merged_segments.append(
                        ShrinkSegment(timestamp, a_min_temp, a_max_temp, b)
                    )
                    self.segments_length += 1
        return merged_segments

    def decompress(self) -> tuple[list[Point], int]:
        "Decompress segments to points."
        start_time = time.time()

        # Pre-calculate the initial timestamps and other values
        init_timestamps: list[int] = [
            segment.get_init_timestamp for segment in self.segments
        ]
        a_values: list[float] = [segment.a for segment in self.segments]
        b_values: list[float] = [segment.get_b for segment in self.segments]
        points: list[Point] = []

        # Loop over segments, avoiding method calls within the loop
        for i in range(len(self.segments) - 1):
            # Calculate the range of timestamps for the current segment
            timestamps = range(init_timestamps[i], init_timestamps[i + 1])
            # Use a list comprehension to generate points
            points += [
                Point(ts, a_values[i] * (ts - init_timestamps[i]) + b_values[i])
                for ts in timestamps
            ]

        # Handle the last segment
        last_segment_timestamps = range(init_timestamps[-1], self.last_timestamp + 1)
        points += [
            Point(ts, a_values[-1] * (ts - init_timestamps[-1]) + b_values[-1])
            for ts in last_segment_timestamps
        ]
        end_time = time.time()
        decompression_base_time = int((end_time - start_time) * 1000)
        return points, decompression_base_time

    def to_bytearray_per_b_segments(
        self, segments: List[ShrinkSegment], variable_byte: bool, out_stream: BytesIO
    ) -> None:
        "Converts b segments to bytearray and writes to file."
        # Initialize a dictionary to organize segments by 'b' value
        input_dict: dict[int, dict[float, list[int]]] = {}
        results_array: list[ShrinkSegment] = []

        if self.epsilon is None:
            raise ValueError("Epsilon must be set!")

        for segment in segments:
            a = segment.get_a
            b = round(segment.get_b / self.epsilon)
            t = segment.get_init_timestamp

            if b not in input_dict:
                input_dict[b] = {}

            if a not in input_dict[b]:
                input_dict[b][a] = []

            input_dict[b][a].append(t)

        # Write the size of the dictionary
        VariableByteEncoder.write(len(input_dict), out_stream)

        if not input_dict.items():
            return

        previous_b: int = min(input_dict.keys())
        VariableByteEncoder.write(previous_b, out_stream)

        for b, a_segments in input_dict.items():
            VariableByteEncoder.write(b - previous_b, out_stream)
            previous_b = b
            VariableByteEncoder.write(len(a_segments), out_stream)

            for a, timestamps in a_segments.items():
                # Custom method to encode the float 'a' value
                FloatEncoder.write(float(a), out_stream)
                len(a_segments)

                if variable_byte:
                    print("variableByte is True, an error occurs\n")
                    timestamps.sort()

                VariableByteEncoder.write(len(timestamps), out_stream)
                previous_timestamp = 0

                for timestamp in timestamps:
                    if variable_byte:
                        print("variableByte is True, an error occurs\n")
                        VariableByteEncoder.write(
                            timestamp - previous_timestamp, out_stream
                        )
                    else:
                        # Custom method to write 'timestamp' as an unsigned int
                        UIntEncoder.write(timestamp, out_stream)
                    previous_timestamp = timestamp
        with open(OUTPUT_PATH_BASE, "w", newline="", encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(results_array)

    def to_byte_array(self, variable_byte: bool, zstd: bool) -> bytes:
        "Convert segments to byte array."
        out_stream = BytesIO()
        out_stream_bytes = None

        if self.epsilon is None:
            raise ValueError("Epsilon must be set!")
        FloatEncoder.write(float(self.epsilon), out_stream)
        self.to_bytearray_per_b_segments(self.segments, variable_byte, out_stream)

        if variable_byte:
            VariableByteEncoder.write(int(self.last_timestamp), out_stream)
        else:
            UIntEncoder.write(self.last_timestamp, out_stream)

        if zstd:
            raise NotImplementedError("zstd is not implemented.")
            # bytes = zstd.compress(outStream.getvalue())
        else:
            out_stream_bytes = out_stream.getvalue()

        out_stream.close()
        return out_stream_bytes

    def save_bytes(self, in_bytes: bytes, filename: str) -> int:
        "Saves bytearray to fiename"
        path = BASE_FOLDER + filename[:-7] + "_base.bin"
        with open(path, "wb") as file:
            file.write(in_bytes)
        base_size = os.path.getsize(path)
        return base_size

    def read_merged_segments_per_b(
        self, variable_byte: bool, in_stream: BytesIO
    ) -> list[ShrinkSegment]:
        "Read byte array that is merged per segment into segments."
        segments: list[ShrinkSegment] = []
        num_b = VariableByteEncoder.read(in_stream)
        timestamp: int = 0

        if num_b == 0:
            return segments

        previous_b = VariableByteEncoder.read(in_stream)

        for _ in range(num_b):
            b = VariableByteEncoder.read(in_stream) + previous_b
            previous_b = b
            num_a = VariableByteEncoder.read(in_stream)

            for _ in range(num_a):
                a = FloatEncoder.read(in_stream)
                num_timestamps = VariableByteEncoder.read(in_stream)
                if self.epsilon is None:
                    raise ValueError("Epsilon must be set!")

                for _ in range(num_timestamps):
                    if variable_byte:
                        timestamp += VariableByteEncoder.read(in_stream)
                    else:
                        timestamp = UIntEncoder.read(in_stream)
                    segments.append(ShrinkSegment(timestamp, a, a, b * self.epsilon))
        return segments

    def read_byte_array(
        self, input_bytes: bytes, variable_byte: bool, zstd: bool
    ) -> None:
        "Read the bytearray to segments."
        if zstd:
            raise NotImplementedError("zstd is not implemented.")
            # binary = zstd.decompress(input_bytes)
        else:
            binary = input_bytes

        in_stream = BytesIO(binary)
        self.epsilon = FloatEncoder.read(in_stream)
        self.segments = self.read_merged_segments_per_b(variable_byte, in_stream)

        if variable_byte:
            self.last_timestamp = VariableByteEncoder.read(in_stream)
        else:
            self.last_timestamp = UIntEncoder.read(in_stream)
        in_stream.close()
