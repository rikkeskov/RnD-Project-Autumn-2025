"""
Data class for the Shrink segment type.
"""

import math


class ShrinkSegment:
    """
    Data class for the Shrink segment type.
    """

    def __init__(self, init_timestamp: int, a_min: float, a_max: float, b: float):
        self.init_timestamp = init_timestamp
        self.a_min = a_min
        self.a_max = a_max
        self.a = (a_min + a_max) / 2
        self.b = b
        self.score = 1  # outlier score

    def round_to_least_decimal(self, a_min: float, a_max: float) -> float:
        "Round to least decimal place."
        a = (a_min + a_max) / 2
        # Determine the number of decimal places in aMin and aMax
        decimal_places_a_min = len(str(a_min).split(".")[1].rstrip("0")) - len(
            str(a_min).split(".")[1].lstrip("0")
        )
        decimal_places_a_max = len(str(a_max).split(".")[1].rstrip("0")) - len(
            str(a_max).split(".")[1].lstrip("0")
        )

        # Find the minimum decimal places to round 'a'
        min_decimal_places = max(decimal_places_a_min, decimal_places_a_max) + 2

        # Round 'a' to the minimum decimal places
        rounded_a = round(a, min_decimal_places)
        return rounded_a

    def count_matching_digits(self, num1: float, num2: float) -> int:
        "Count matching digits."
        if math.isinf(num1) or math.isinf(num1):
            return -1
        leading_num1 = str(num1).split(".", maxsplit=1)[0]
        leading_num2 = str(num1).split(".", maxsplit=1)[0]

        str_num1 = str(num1).split(".")[1]  # Get the number after the decimal point
        str_num2 = str(num2).split(".")[1]  # Get the number after the decimal point
        count = 0
        for digit1, digit2 in zip(str_num1, str_num2):
            if digit1 == digit2:
                count += 1
            else:
                break

        leading_num1 = str(num1).split(".", maxsplit=1)[0]
        leading_num2 = str(num1).split(".", maxsplit=1)[0]
        if leading_num1 != leading_num2:
            return 0
        return count

    def truncate_to_n_decimal_places(self, number: float, n: int) -> float:
        "Enlarge to n decimal places."
        scaled_number = number * (
            10**n
        )  # Enlarge the floating point number by n digits
        # Convert the enlarged number back to its original size and truncate the decimal part
        truncated_number = int(scaled_number) / (10**n)
        return truncated_number

    @property
    def get_init_timestamp(self) -> int:
        "Initial timestamp."
        return self.init_timestamp

    @property
    def get_a_min(self) -> float:
        "Minimum of a."
        return self.a_min

    @property
    def get_a_max(self) -> float:
        "Maximum of a."
        return self.a_max

    @property
    def get_a(self) -> float:
        "A in itself."
        if (self.a_max * self.a_min) < 0:
            self.a = 0
            return self.a
        precision = self.count_matching_digits(self.a_max, self.a_min)
        if precision == -1:
            self.a = 0
            return self.a
        if precision < 2:
            precision = 2
        a = self.truncate_to_n_decimal_places(
            (self.a_max + self.a_min) / 2, precision + 2
        )

        if not (a >= self.a_min and a <= self.a_max):
            a = self.round_to_least_decimal(self.a_max, self.a_min)

        if not (a >= self.a_min and a <= self.a_max):
            a = self.truncate_to_n_decimal_places(
                (self.a_max + self.a_min) / 2, precision + 3
            )

        self.a = a
        if not (a >= self.a_min and a <= self.a_max):
            self.a = (self.a_max + self.a_min) / 2

        leading_num1 = str(self.a_max).split(".", maxsplit=1)[0]
        leading_num2 = str(self.a_min).split(".", maxsplit=1)[0]
        if leading_num1 != leading_num2:
            self.a = round((self.a_max + self.a_min) / 2, 1)
            return self.a

        if abs(self.a) < 1e-4:
            # This needs to be noted to see if it will cause problems!!!
            self.a = 0
        # print("self.a_max = ", self.a_max, "  self.a_min = ", self.a_min, "  self.a = ", self.a)
        return self.a

    @property
    def get_b(self) -> float:
        "Get actual b."
        return self.b

    def show(self) -> None:
        "Print method."
        print("[ ", self.get_init_timestamp, ", ", self.a, ", ", self.b, "]")


if __name__ == "__main__":
    # Example usage of the class
    segment = ShrinkSegment(123456789, 0.1, 0.2, 5.0)
    print(segment.get_init_timestamp)
    print(segment.get_a_min)
    print(segment.get_a_max)
    print(segment.get_a)
    print(segment.get_b)
