import torch
from pyblaz.compression import PyBlaz
import numpy as np

import math


def compression_ratio(
    original_shape: tuple,
    original_element_size: int,
    block_shape: tuple,
    keep_proportion: float,
    float_size: int,
    int_size: int,
):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
codec = PyBlaz(block_shape=(8, 8), dtype=torch.float32, index_dtype=torch.int16, device=device)

x_train_0 = np.fromfile("../../original_dat_128X128_size1000/x_train_0_128X128_size1000.dat", dtype=np.float32)
x_train_0 = x_train_0.reshape(1000, 128)
x_train_1 = np.fromfile("../../original_dat_128X128_size1000/x_train_1_128X128_size1000.dat", dtype=np.float32)
x_train_1 = x_train_1.reshape(16384, 2)
x_test_0 = np.fromfile("../../original_dat_128X128_size1000/x_test_0_128X128_size1000.dat", dtype=np.float32)
x_test_0 = x_test_0.reshape(1000, 128)
x_test_1 = np.fromfile("../../original_dat_128X128_size1000/x_test_1_128X128_size1000.dat", dtype=np.float32)
x_test_1 = x_test_1.reshape(16384, 2)

y_train = np.fromfile("../../original_dat_128X128_size1000/y_train_128X128_size1000.dat", dtype=np.float32)
y_train = y_train.reshape(1000, 16384)
y_test = np.fromfile("../../original_dat_128X128_size1000/y_test_128X128_size1000.dat", dtype=np.float32)
y_test = y_test.reshape(1000, 16384)

compressed_x_train_0 = codec.compress(torch.tensor(x_train_0, dtype=torch.float32, device=device))
decompressed_x_train_0 = (codec.decompress(compressed_x_train_0)).cpu().numpy()

compressed_x_train_1 = codec.compress(torch.tensor(x_train_1, dtype=torch.float32, device=device))
decompressed_x_train_1 = (codec.decompress(compressed_x_train_1)).cpu().numpy()

compressed_x_test_0 = codec.compress(torch.tensor(x_test_0, dtype=torch.float32, device=device))
decompressed_x_test_0 = (codec.decompress(compressed_x_test_0)).cpu().numpy()

compressed_x_test_1 = codec.compress(torch.tensor(x_test_1, dtype=torch.float32, device=device))
decompressed_x_test_1 = (codec.decompress(compressed_x_test_1)).cpu().numpy()

compressed_y_train = codec.compress(torch.tensor(y_train, dtype=torch.float32, device=device))
decompressed_y_train = (codec.decompress(compressed_y_train)).cpu().numpy()

compressed_y_test = codec.compress(torch.tensor(y_test, dtype=torch.float32, device=device))
decompressed_y_test = (codec.decompress(compressed_y_test)).cpu().numpy()

decompressed_x_train_0.tofile(
    "../../original_dat_128X128_size1000/pyBlaz/blocksize_8_index_16/decompressed_dat/x_train_0.dat"
)
decompressed_x_train_1.tofile(
    "../../original_dat_128X128_size1000/pyBlaz/blocksize_8_index_16/decompressed_dat/x_train_1.dat"
)
decompressed_x_test_0.tofile(
    "../../original_dat_128X128_size1000/pyBlaz/blocksize_8_index_16/decompressed_dat/x_test_0.dat"
)
decompressed_x_test_1.tofile(
    "../../original_dat_128X128_size1000/pyBlaz/blocksize_8_index_16/decompressed_dat/x_test_1.dat"
)
decompressed_y_train.tofile(
    "../../original_dat_128X128_size1000/pyBlaz/blocksize_8_index_16/decompressed_dat/y_train.dat"
)
decompressed_y_test.tofile(
    "../../original_dat_128X128_size1000/pyBlaz/blocksize_8_index_16/decompressed_dat/y_test.dat"
)
