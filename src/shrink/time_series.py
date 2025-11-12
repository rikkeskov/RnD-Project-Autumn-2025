"""
Data class for time series format.
"""

from .point import Point


class TimeSeries:
    "Time series data class."

    def __init__(self, data: list[Point], data_range: float):
        self.data = data  # Points
        self.data_range = data_range
        self.size = len(data) * (4 + 4)

    def length(self) -> int:
        "Length of data"
        return len(self.data)
