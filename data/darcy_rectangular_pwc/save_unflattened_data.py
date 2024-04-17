import argparse
import pathlib

import scipy.io
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-folder", type=str, default="unflattened_data")
    args = parser.parse_args()

    n_train = 1024
    n_test = 1024

    data_folder = pathlib.Path("/datasets") / "lulu_cmame2022" / "Darcy_rectangular_PWC"
    save_folder = pathlib.Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    x_train, y_train = get_data(data_folder / "piececonst_r421_N1024_smooth1.mat", n_train, resolution=1)
    x_test, y_test = get_data(data_folder / "piececonst_r421_N1024_smooth2.mat", n_test, resolution=1)

    # Arrays have been flattened to x_train[0]: (n_train, 421 * 421), x_train[1]: (421 * 421, 2), y_train: (n_train, 421 * 421)
    size = 421
    x_train = (x_train[0].reshape(n_train, size, size), x_train[1].reshape(size, size, 2))
    x_test = (x_test[0].reshape(n_test, size, size), x_test[1].reshape(size, size, 2))
    y_train = y_train.reshape(n_train, size, size)
    y_test = y_test.reshape(n_test, size, size)

    assert all(
        (
            x_train[0].dtype == np.float32,
            x_train[1].dtype == np.float32,
            y_train.dtype == np.float32,
            x_test[0].dtype == np.float32,
            x_test[1].dtype == np.float32,
            y_test.dtype == np.float32,
        )
    ), "Data types should be float32."

    x_train[0].tofile(save_folder / "x_train_branch_unflattened.bin")
    x_train[1].tofile(save_folder / "x_train_trunk_unflattened.bin")
    y_train.tofile(save_folder / "y_train_unflattened.bin")
    x_test[0].tofile(save_folder / "x_test_branch_unflattened.bin")
    x_test[1].tofile(save_folder / "x_test_trunk_unflattened.bin")
    y_test.tofile(save_folder / "y_test_unflattened.bin")

    print("Don't forget to flatten them again before using them in the model.")


def get_data(filename, ndata, resolution):
    size = int(((421 - 1) / resolution) + 1)
    data = scipy.io.loadmat(filename)
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
