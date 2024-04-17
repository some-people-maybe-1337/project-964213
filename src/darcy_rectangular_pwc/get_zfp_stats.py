import sys
import argparse
import pathlib
import pickle

import numpy as np
from scipy import io
import zfpy

# import tensorflow_addons as tfa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=1)
    args = parser.parse_args()

    n_train = 1024  # up to 1024
    n_test = 256  # up to 1024

    data_folder = pathlib.Path("/datasets") / "lulu_cmame2022" / "Darcy_rectangular_PWC"

    x_train, y_train = get_data(data_folder / "piececonst_r421_N1024_smooth1.mat", n_train, resolution=args.resolution)
    x_test, y_test = get_data(data_folder / "piececonst_r421_N1024_smooth2.mat", n_test, resolution=args.resolution)

    x_train_branch, x_train_trunk = x_train
    x_test_branch, x_test_trunk = x_test

    x_train_branch_mean = np.mean(x_train_branch)
    x_train_branch_std = np.std(x_train_branch)

    x_test_branch_mean = np.mean(x_test_branch)
    x_test_branch_std = np.std(x_test_branch)

    y_train_mean = np.mean(y_train)
    y_train_std = np.std(y_train)

    y_test_mean = np.mean(y_test)
    y_test_std = np.std(y_test)

    print(f"Train branch mean: {x_train_branch_mean}, std: {x_train_branch_std}")
    print(f"Test branch mean: {x_test_branch_mean}, std: {x_test_branch_std}")
    print(f"Train trunk mean: {np.mean(x_train_trunk)}, std: {np.std(x_train_trunk)}")
    print(f"Test trunk mean: {np.mean(x_test_trunk)}, std: {np.std(x_test_trunk)}")
    print(f"Train label mean: {y_train_mean}, std: {y_train_std}")
    print(f"Test label mean: {y_test_mean}, std: {y_test_std}")

    to_write = ["std multiplier,train branch,test branch,train label"]
    for std_multiplier in (0.1, 0.01, 0.001, 0.0001):
        size = int(((421 - 1) / args.resolution) + 1)
        x_train_branch = x_train[0].reshape(-1, size, size)
        x_test_branch = x_test[0].reshape(-1, size, size)
        y_train = y_train.reshape(-1, size, size)
        # Shouldn't perturb test labels.

        x_train_branch_compressed = zfpy.compress_numpy(x_train_branch, tolerance=x_train_branch_std * std_multiplier)
        x_test_branch_compressed = zfpy.compress_numpy(x_test_branch, tolerance=x_test_branch_std * std_multiplier)
        y_train_compressed = zfpy.compress_numpy(y_train, tolerance=y_train_std * std_multiplier)

        to_write.append(
            f"{std_multiplier},"
            f"{compression_ratio(x_train_branch, x_train_branch_compressed)},"
            f"{compression_ratio(x_test_branch, x_test_branch_compressed)},"
            f"{compression_ratio(y_train, y_train_compressed)}"
        )

    with open("zfp_ratios.csv", "w") as f:
        f.write("\n".join(to_write))


def compression_ratio(original: np.ndarray, compressed: bytes):
    return original.nbytes / len(compressed)


def get_data(filename, ndata, resolution):
    """
    Shapes x[0]: (ndata, 421 * 421), x[1]: (421 * 421, 2), y: (ndata, 421 * 421)
    """
    size = int(((421 - 1) / resolution) + 1)
    data = io.loadmat(filename)
    x_branch = data["coeff"][:ndata, ::resolution, ::resolution].astype(np.float32) * 0.1 - 0.75
    y = data["sol"][:ndata, ::resolution, ::resolution].astype(np.float32) * 100
    # The dataset has a mistake that the BC is not 0.
    y[:, 0, :] = 0
    y[:, -1, :] = 0
    y[:, :, 0] = 0
    y[:, :, -1] = 0

    grids = []
    grids.append(np.linspace(0, 1, size, dtype=np.float32))
    grids.append(np.linspace(0, 1, size, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    x_branch = x_branch.reshape(ndata, size * size)
    x = (x_branch, grid)
    y = y.reshape(ndata, size * size)
    return x, y


if __name__ == "__main__":
    main()
