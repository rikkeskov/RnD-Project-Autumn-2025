"""Module containing class for unsigned integer to bytes encoder."""

from io import BytesIO
import struct


class UIntEncoder:
    """
    Unsigned integer to bytes encoder.
    """

    @staticmethod
    def write(number: int, output_stream: BytesIO) -> None:
        "Write unsigned integer to bytes."
        if number > (2**32) - 1 or number < 0:
            raise ValueError(f"Can't save number {number} as unsigned int")
        int_bytes = struct.pack(">I", number)
        output_stream.write(int_bytes)

    @staticmethod
    def read(input_stream: BytesIO) -> int:
        "Read bytes as unsigned integer."
        int_bytes = input_stream.read(4)
        if len(int_bytes) != 4:
            raise ValueError("Invalid byte length")
        number: int = struct.unpack(">I", int_bytes)[0]
        return number
