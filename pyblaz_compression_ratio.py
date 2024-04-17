import math


def compression_ratio(
    original_shape: tuple,
    original_element_size: int,
    block_shape: tuple,
    keep_proportion: float,
    float_size: int,
    int_size: int,
):
    """
    Returns the compression ratio of a tensor compressed using the specified parameters.

    Args:
        original_shape: Shape of the original tensor.
        original_element_size: Size of each element in the original tensor.
        block_shape: Shape of the blocks used for compression.
        keep_proportion: Proportion of specified coefficients to total number of coefficients.
        float_size: Number of bits used to store each floating point number.
        int_size: Number of bits used to store each integer number.
    """
    return uncompressed_size(original_shape, original_element_size) / compressed_size(
        original_shape, block_shape, keep_proportion, float_size, int_size
    )


def compressed_size(
    original_shape: tuple,
    block_shape: tuple,
    keep_proportion: float,
    float_size: int,
    int_size: int,
):
    n_blocks = math.prod(
        math.ceil(tensor_size / block_size) for tensor_size, block_size in zip(original_shape, block_shape)
    )
    return (
        4
        + 64 * len(original_shape)
        + 64
        + 64 * len(original_shape)
        + math.prod(block_shape)
        + float_size * n_blocks
        + int_size * keep_proportion * n_blocks * math.prod(block_shape)
    )


def uncompressed_size(original_shape: tuple, element_size: int):
    return element_size * len(original_shape) + 64 + element_size * math.prod(original_shape)
