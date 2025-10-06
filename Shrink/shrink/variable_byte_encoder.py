"""
Variable byte encode for compression.
"""
from io import BytesIO
import random

class VariableByteEncoder:
    """
    Variable byte encode for compression.
    """
    @staticmethod
    def extract7bits(i: int, val: int) -> int:
        "Extract 7 bits."
        return (val >> (7 * i)) & ((1 << 7) - 1)
    
    @staticmethod
    def extract7bitsmaskless(i: int, val: int) -> int:
        "Extract 7 bits."
        result = (val >> (7 * i)) & 0xFF  # Ensure the result is in the range 0-255
        if result & 0x80:  # If the highest bit is 1, it means a negative number, and the sign is extended
            result = result - 256  # Sign extension using the signed right shift operator
        return result
    
    @staticmethod
    def write(num: int, out_stream: BytesIO) -> None:
        "Write number to bytes."
        if num < 0:
            num = ((-num) << 1) | 1  # Use Zigzag encoding to handle negative numbers.
        else:
            num = num << 1  # Handling positive numbers.

        while num >= 128:
            out_stream.write(bytes([num & 0x7F | 0x80]))
            num >>= 7

        # For positive numbers the lowest bit is set to 0. There is no need to set highest bit to 1.
        out_stream.write(bytes([num]))  

    @staticmethod
    def read(in_stream: BytesIO) -> int:
        "Read bytes to number."
        num: int = 0
        shift = 0

        while True:
            inp = in_stream.read(1)
            if not inp:
                break
            inp = inp[0]
            num |= (inp & 0x7F) << shift
            shift += 7
            if not inp & 0x80:
                break

        if num & 1:
            num = -(num >> 1)  # Restore negative numbers
        else:
            num = num >> 1
        return num


def test_variable_byte_encoder():
    "Test variable byte encoder on random data."
    random_numbers = [random.randint(-1000000, 1000000) for _ in range(10)]

    for num in random_numbers:
        out_stream = BytesIO()
        VariableByteEncoder.write(num, out_stream)

        input_stream = BytesIO(out_stream.getvalue())
        decoded_number = VariableByteEncoder.read(input_stream)

        print(f"Original number: {num}")
        print(f"Decoded number: {decoded_number}")
        print(f"Is equal: {num == decoded_number}")


if __name__=='__main__':
    numbers = [42, 128, -1024, 65535, 1000000]
    for number in numbers:
        output_stream = BytesIO()
        VariableByteEncoder.write(number, output_stream)
        encoded_bytes = output_stream.getvalue()

        print(f"Original number: {number}")
        print(f"Encoded bytes: {encoded_bytes}")
        print(f"Encoded length: {len(encoded_bytes)} bytes")
