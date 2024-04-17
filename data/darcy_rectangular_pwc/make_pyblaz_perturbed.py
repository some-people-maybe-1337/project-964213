import pathlib
import argparse

import scipy.io
import numpy as np
import torch
import tqdm

from pyblaz.compression import PyBlaz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-folder", type=str, default="pyblaz_perturbed_data")
    parser.add_argument("--resolution", type=int, default=1)
    args = parser.parse_args()

    n_train = 1024  # up to 1024
    n_test = 256  # up to 1024

    data_folder = pathlib.Path("/datasets") / "lulu_cmame2022" / "Darcy_rectangular_PWC"
    x_train, y_train = get_data(data_folder / "piececonst_r421_N1024_smooth1.mat", n_train, resolution=args.resolution)
    x_test, y_test = get_data(data_folder / "piececonst_r421_N1024_smooth2.mat", n_test, resolution=args.resolution)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Arrays have been flattened to x_train[0]: (n_train, 421 * 421), x_train[1]: (421 * 421, 2), y_train: (n_train, 421 * 421)
    # modulo resolution.
    # PyBlaz will make better sense of it if unflattened.
    size = int(((421 - 1) / args.resolution) + 1)
    x_train = (x_train[0].reshape(n_train, size, size), x_train[1].reshape(size, size, 2))
    x_test = (x_test[0].reshape(n_test, size, size), x_test[1].reshape(size, size, 2))
    y_train = y_train.reshape(n_train, size, size)
    y_test = y_test.reshape(n_test, size, size)

    block_sizes = (4, 8)
    float_types = (torch.float32,)
    index_dtypes = (torch.int8, torch.int16, torch.int32)

    progress_bar = tqdm.tqdm(total=len(block_sizes) * len(float_types) * len(index_dtypes))
    for block_size in block_sizes:
        for float_type in float_types:
            for index_dtype in index_dtypes:
                save_folder = (
                    pathlib.Path(args.save_folder)
                    / f"r{args.resolution}"
                    / f"bs{block_size}_{str(float_type)[6:]}_{str(index_dtype)[6:]}"
                )
                save_folder.mkdir(parents=True, exist_ok=True)

                codec = PyBlaz(
                    block_shape=(1, block_size, block_size), dtype=float_type, index_dtype=index_dtype, device=device
                )

                x_train_branch_perturbed = (
                    codec.decompress(codec.compress(torch.tensor(x_train[0], dtype=float_type, device=device)))
                    .reshape(n_train, size * size)
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                x_train_trunk_perturbed = (
                    codec.decompress(
                        codec.compress(torch.tensor(x_train[1], dtype=float_type, device=device).permute(2, 0, 1))
                    )
                    .permute(1, 2, 0)
                    .reshape(size * size, 2)
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                y_train_perturbed = (
                    codec.decompress(codec.compress(torch.tensor(y_train, dtype=float_type, device=device)))
                    .reshape(n_train, size * size)
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )

                x_test_branch_perturbed = (
                    codec.decompress(codec.compress(torch.tensor(x_test[0], dtype=float_type, device=device)))
                    .reshape(n_test, size * size)
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                x_test_trunk_perturbed = (
                    codec.decompress(
                        codec.compress(torch.tensor(x_test[1], dtype=float_type, device=device).permute(2, 0, 1))
                    )
                    .permute(1, 2, 0)
                    .reshape(size * size, 2)
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                y_test_perturbed = (
                    codec.decompress(codec.compress(torch.tensor(y_test, dtype=float_type, device=device)))
                    .reshape(n_test, size * size)
                    .to(torch.float32)
                    .cpu()
                    .numpy()
                )
                np.save(save_folder / "x_train_branch.npy", x_train_branch_perturbed)
                np.save(save_folder / "x_train_trunk.npy", x_train_trunk_perturbed)
                np.save(save_folder / "y_train.npy", y_train_perturbed)
                np.save(save_folder / "x_test_branch.npy", x_test_branch_perturbed)
                np.save(save_folder / "x_test_trunk.npy", x_test_trunk_perturbed)
                np.save(save_folder / "y_test.npy", y_test_perturbed)

                progress_bar.write(
                    f"{np.max(np.abs((x_train_branch_perturbed - x_train[0].reshape(n_train, size * size))))},"
                    f"{np.max(np.abs((x_train_trunk_perturbed - x_train[1].reshape(size * size, 2))))},"
                    f"{np.max(np.abs((y_train_perturbed - y_train.reshape(n_train, size * size))))},"
                    f"{np.max(np.abs((x_test_branch_perturbed - x_test[0].reshape(n_test, size * size))))},"
                    f"{np.max(np.abs((x_test_trunk_perturbed - x_test[1].reshape(size * size, 2))))},"
                    f"{np.max(np.abs((y_test_perturbed - y_test.reshape(n_test, size * size))))}"
                )
                progress_bar.update()


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
