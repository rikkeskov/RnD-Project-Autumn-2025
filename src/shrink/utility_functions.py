"""
Utility functions.
"""

from decimal import Decimal
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .point import Point


def plot_line_graph(
    x: list[float | int],
    y: list[float | int],
    title: str = "Two Line Graphs on One Plot",
    color: str = "blue",
) -> None:
    """
    Draws a line chart.
    Parameters:
    x, y: corresponds to the x-axis and y-axis
    """
    plt.figure(figsize=(10, 6))  # type: ignore
    plt.plot(x, y, marker="o", color=color)  # type: ignore
    plt.title(title)  # type: ignore
    plt.xlabel("Timestamp")  # type: ignore
    plt.ylabel("Value")  # type: ignore
    plt.grid(True)  # type: ignore

    # Annotate each point with its value
    for i, value in enumerate(y):
        plt.text(x[i], y[i], f"{value:.4f}", ha="center", va="bottom")  # type: ignore
    plt.show()  # type: ignore


def plot_simpleline(
    points: list[Point] | None = None,
    title: str = "Two Line Graphs on One Plot",
    color: str = "blue",
) -> None:
    """
    Draws a line chart.
    Parameters:
    x, y: corresponds to the x-axis and y-axis
    """
    if points is None:
        return None
    # Extract timestamps and values for plotting.

    timestamps = [point.timestamp for point in points]
    values = [point.value for point in points]

    # Draw a line chart (without points)
    plt.figure(figsize=(10, 6))  # type: ignore
    # plt.plot(x, y)
    plt.plot(timestamps, values, color=color)  # type: ignore
    plt.title(title)  # type: ignore
    plt.xlabel("Timestamp")  # type: ignore
    plt.ylabel("Value")  # type: ignore
    plt.grid(False)  # type: ignore
    plt.tick_params(axis="x", which="both", labelbottom=False)  # type: ignore
    plt.show()  # type: ignore
    return None


def plot_two_line_graphs(
    points1: list[Point],
    points2: list[Point],
    end: int = 100,
    title: str = "Two Line Graphs on One Plot",
) -> None:
    """
    Draw two line charts on one graph.
    Parameters:
    points1 (list of Point): List of Point objects for the first line chart
    points2 (list of Point): List of Point objects for the second line chart
    """
    # Extract the timestamp and value of the first line graph
    timestamps1 = [point.timestamp for point in points1]
    values1 = [point.value for point in points1]

    # Extract the timestamp and value of the second line graph
    timestamps2 = [point.timestamp for point in points2]
    values2 = [point.value for point in points2]

    # Draw two line charts
    plt.figure(figsize=(14, 10))  # type: ignore
    plt.plot(timestamps1[:end], values1[:end], label="Original data", color="red")  # type: ignore
    plt.plot(timestamps2[:end], values2[:end], label="Approximated data", color="green")  # type: ignore
    plt.title(title)  # type: ignore
    plt.xlabel("Timestamp")  # type: ignore
    plt.ylabel("Value")  # type: ignore
    plt.legend()  # type: ignore
    plt.grid(True)  # type: ignore

    # Annotate each point with its value
    for i, value in enumerate(values2):
        plt.text(timestamps2[i], values2[i], f"{value:.1f}", ha="center", va="bottom")  # type: ignore
    plt.show()  # type: ignore


def piecewise_constant_approximation(data: list[float], num_segments: int = 1):
    "Piecewise constant approximation."
    # Calculate the length of each segment
    segment_length = len(data) // num_segments

    # Pre-allocate the array for performance
    approximated_data = np.zeros_like(data)
    # Compute the approximation
    for i in range(num_segments):
        start_index = i * segment_length
        # Handle the last segment which might have different length
        end_index = len(data) if i == num_segments - 1 else start_index + segment_length
        # Compute the mean of the segment
        segment_mean = np.mean(data[start_index:end_index])
        # Assign the mean to the segment's data points
        approximated_data[start_index:end_index] = segment_mean

    return approximated_data


def generated_list(low: float = -1, high: float = 1, size: int = 10):
    "Generate normalised list."
    # Generate a list of floats with a mean of 0 and a uniform distribution between -2 and 2
    gen_list = np.random.uniform(low, high, size)

    # Adjust the list to have a mean of 0
    adjusted_list = gen_list - np.mean(gen_list)

    # Ensure that the maximum and minimum values are within the specified range
    adjusted_list = np.clip(adjusted_list, low, high)
    return adjusted_list


def float_to_int(float_array: np.ndarray[Any, Any], error_threshold: float = 1):
    "Float to int with precision."
    # Convert the input to a numpy array if it's not already one
    float_array = np.array(float_array)
    # Multiply by 20, round, and then divide by 20 to ensure the error is within 0.05
    int_array = np.round(float_array / error_threshold).astype(int)
    return int_array


def int_to_float(int_array: np.ndarray[Any, Any], error_threshold: float = 1):
    "Integer to float array"
    # Convert back to float
    float_array = int_array * error_threshold
    return float_array


def compress_ratio_for_int(int_array: list[int]):
    "Compression ratio for integer array."
    # Original size in bits (16 bits per int)
    original_size = len(int_array) * 32

    # Calculate the size after compression
    # (sum of the bit lengths of each int's binary representation)
    # For the number 0, we count 1 bit.
    compressed_size = sum(
        len(bin(abs(i)).lstrip("-0b")) if i != 0 else 1 for i in int_array
    )

    # Calculate the compression ratio
    compression_ratio = original_size / compressed_size if compressed_size != 0 else 0
    return compression_ratio


def totalbits_when_int(int_array: list[int]):
    "total bits for the integer array."
    # Calculate the total number of bits after conversion (account for the number 0 as 1 bit)
    total_compressed_bits = sum(
        len(bin(abs(i)).lstrip("-0b")) if i != 0 else 1 for i in int_array
    )
    return total_compressed_bits


def quantize_values(datas: np.ndarray[Any, Any], error: float):
    "Quantize values in datas within error bound."
    max_val = max(datas)
    min_val = min(datas)
    rounded_res = [round((v - min_val) / (max_val - min_val) / error) for v in datas]
    return np.array(rounded_res), max_val, min_val


def dequantize_values(
    quan_datas: np.ndarray[Any, Any], max_val: float, min_val: float, error: float
):
    "Dequantize the values."
    return [v * (max_val - min_val) * error + min_val for v in quan_datas]


def res_quantize(x: np.ndarray[Any, Any], epsilon: float):
    "Quantize the result."
    res = [round((v / epsilon)) for v in x]
    return res


def res_dequantize(x_quant: list[float], epsilon: float):
    "De quantize the result."
    return [v * epsilon for v in x_quant]


def high_precision_add(x: float, y: float):
    "High precision addition."
    return float(Decimal(str(x)) + Decimal(str(y)))


def high_precision_subtract(x: float, y: float):
    "High precision subtraction."
    return float(Decimal(str(x)) - Decimal(str(y)))


def get_precision(num: float):
    "Get number precision."
    return len(str(num).split(".")[1])


if __name__ == "__main__":
    generated_list_result = generated_list(-1.145, 1.145, 20)
    print("Original List = ", generated_list_result)
    intList = float_to_int(generated_list_result, error_threshold=0.05)
    print("Transformed int list = ", float_to_int(intList))
    approxiList = int_to_float(intList, error_threshold=0.05)
    print("Float list = ", approxiList)
    print("error is = ", generated_list_result - approxiList)

    generated_list_result = generated_list(-1.145, 1.145, 5)
    print(generated_list_result)
    quantized_datas, maxval, minval = quantize_values(generated_list_result, 0.05)
    print(quantized_datas)
    dequanized_data = dequantize_values(quantized_datas, maxval, minval, 0.05)
    print(dequanized_data)
