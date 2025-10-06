"""
Module for data class Point.
"""

class Point:
    """
    Data class. Point in each time series.
    """
    def __init__(self, timestamp: int, value: float):
        self._timestamp = timestamp
        self._value = value

    @property
    def timestamp(self) -> int:
        "Timestamp for point in time series."
        return self._timestamp

    @property
    def value(self) -> float:
        "Value for point in time series."
        return self._value

    def add(self, val: float):
        "Add to value."
        self._value += val

    def set(self, val: float):
        "Set value."
        self._value = val
