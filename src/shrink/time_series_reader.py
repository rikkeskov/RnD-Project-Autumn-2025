"""Module for time series data type reader."""

import os
import csv

from .point import Point
from .time_series import TimeSeries


class TimeSeriesReader:
    """Ready for time series data type."""

    @staticmethod
    def get_time_series(csv_file: str) -> TimeSeries:
        """Get time series."""
        points: list[Point] = []
        max_val = float("-inf")
        min_val = float("inf")

        try:
            with open(csv_file, "r", newline="", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    timestamp = int(row[0])
                    try:
                        value = float(row[1])
                    except ValueError as e:
                        print(f"Error: {e} for row ID {timestamp} with value {row[1]}")
                        continue
                    points.append(Point(timestamp, value))
                    max_val = max(max_val, value)
                    min_val = min(min_val, value)
        except OSError as e:
            print(e)
            raise OSError("See print.") from e
        ts = TimeSeries(points, max_val - min_val)

        ## use the size of .csv. If using bit size, comment this.
        ts.size = os.path.getsize(csv_file)
        return ts
