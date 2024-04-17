import argparse
import pathlib

import scipy.io
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tqdm
import zfpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-folder", type=str, default="figures/perturbation_analysis")
    parser.add_argument("--resolution", "-r", type=int, default=1)
    parser.add_argument(
        "--pyblaz-perturbed-folder",
        type=str,
        default="../../data/darcy_rectangular_pwc/pyblaz_perturbed_data",
    )
    parser.add_argument("--sz_perturbed_folder", type=str, default="../../data/darcy_rectangular_pwc/sz_decompressed")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--figure-format", type=str, default="pdf")
    args = parser.parse_args()

    n_train = 1024
    n_test = 256

    data_folder = pathlib.Path("/datasets") / "lulu_cmame2022" / "Darcy_rectangular_PWC"

    print("Reading train data. Might take a while.")
    x_train, y_train = get_data(data_folder / "piececonst_r421_N1024_smooth1.mat", n_train, resolution=1)
    print("Reading test data. Might take a while.")
    x_test, y_test = get_data(data_folder / "piececonst_r421_N1024_smooth2.mat", n_test, resolution=1)

    x_train_branch = x_train[0]
    # x_train_trunk = x_train[1]
    x_test_branch = x_test[0]
    # x_test_trunk = x_test[1]
    y_train = y_train
    y_test = y_test
    # Shapes: x_branch: (n, 421 * 421), x_trunk: (421 * 421, 2), y: (n, 421 * 421)

    pyblaz_perturbations = gather_pyblaz_perturbations(args, y_train, y_test, x_train_branch, x_test_branch)
    plot_sz_and_zfp_histograms(
        args, y_train, y_test, x_train_branch, x_test_branch, bins=args.bins, pyblaz_perturbations=pyblaz_perturbations
    )


def plot_sz_and_zfp_histograms(
    args, y_train, y_test, x_train_branch, x_test_branch, bins=100, pyblaz_perturbations=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = int(((421 - 1) / args.resolution) + 1)

    subplot_size = 3
    sequential_colormap = "viridis"
    diverging_colormap = "seismic"

    sz_perturbed_folder = pathlib.Path(args.sz_perturbed_folder)
    combined_save_folder = pathlib.Path(args.save_folder) / f"r{args.resolution}" / "combined"
    fft_save_folder = pathlib.Path(args.save_folder) / f"r{args.resolution}" / "fft"
    combined_save_folder.mkdir(parents=True, exist_ok=True)
    fft_save_folder.mkdir(parents=True, exist_ok=True)

    pretty_subset_names = {
        "x_train_branch": "train $u(x)$",
        "x_test_branch": "test $u(x)$",
        "y_train": "train $v(y)$",
        "y_test": "test $v(y)$",
    }

    progress_bar = tqdm.tqdm(total=4 * 4, desc="Plotting")
    for data_type in [
        ("x_train_branch", x_train_branch),
        ("x_test_branch", x_test_branch),
        ("y_train", y_train),
        ("y_test", y_test),
    ]:
        for std_multiplier in (0.1, 0.01, 0.001, 0.0001):
            sz_perturbed_data = read_binary_and_reshape(
                sz_perturbed_folder / f"{data_type[0]}_{std_multiplier}.bin", size
            )
            zfp_perturbed_data = zfpy.decompress_numpy(
                zfpy.compress_numpy(data_type[1].reshape(-1, size, size), tolerance=data_type[1].std() * std_multiplier)
            ).reshape(-1, size * size)

            sz_perturbation = sz_perturbed_data - data_type[1]
            zfp_perturbation = zfp_perturbed_data - data_type[1]

            abs_range = max(np.abs(sz_perturbation).max(), np.abs(zfp_perturbation).max())

            sz_worst_index = np.abs(sz_perturbation).max(axis=1).argmax()
            zfp_worst_index = np.abs(zfp_perturbation).max(axis=1).argmax()

            perturbations = [sz_perturbation, zfp_perturbation]
            labels = ["SZ", "ZFP"]
            if pyblaz_perturbations is not None:
                closest_pyblaz_perturbation_name, closest_pyblaz_perturbation = min(
                    ((key, perturbation[data_type[0]]) for key, perturbation in pyblaz_perturbations.items()),
                    key=lambda pair: abs(abs_range - np.abs(pair[1]).max()),
                )
                perturbations.append(closest_pyblaz_perturbation)
                labels.append(make_pyblaz_label(closest_pyblaz_perturbation_name))
                pyblaz_worst_index = np.abs(closest_pyblaz_perturbation).max(axis=1).argmax()

            vmax = max(np.abs(sz_perturbed_data).max(), np.abs(zfp_perturbed_data).max())
            if pyblaz_perturbations is not None:
                vmax = max(vmax, np.abs(closest_pyblaz_perturbation).max())
            vmin = -vmax
            fig, axs = plt.subplots(
                2 + bool(pyblaz_perturbations),
                3,
                figsize=(subplot_size * 3 + 6, subplot_size * 3),
                constrained_layout=True,
            )

            cax1 = axs[0, 0].imshow(
                data_type[1][sz_worst_index].reshape(size, size),
                cmap=sequential_colormap if "y" in data_type[0] else diverging_colormap,
                vmin=0 if "y" in data_type[0] else vmin,
                vmax=vmax,
            )
            axs[0, 0].set_title("original")
            axs[0, 0].set_xticks([])
            axs[0, 0].set_yticks([])

            axs[0, 1].imshow(
                sz_perturbed_data[sz_worst_index].reshape(size, size),
                cmap=sequential_colormap if "y" in data_type[0] else diverging_colormap,
                vmin=0 if "y" in data_type[0] else vmin,
                vmax=vmax,
            )
            axs[0, 1].set_title("SZ-perturbed")
            axs[0, 1].set_xticks([])
            axs[0, 1].set_yticks([])

            axs[1, 0].imshow(
                data_type[1][zfp_worst_index].reshape(size, size),
                cmap=sequential_colormap if "y" in data_type[0] else diverging_colormap,
                vmin=0 if "y" in data_type[0] else vmin,
                vmax=vmax,
            )
            axs[1, 0].set_title("original")
            axs[1, 0].set_xticks([])
            axs[1, 0].set_yticks([])

            axs[1, 1].imshow(
                zfp_perturbed_data[zfp_worst_index].reshape(size, size),
                cmap=sequential_colormap if "y" in data_type[0] else diverging_colormap,
                vmin=0 if "y" in data_type[0] else vmin,
                vmax=vmax,
            )
            axs[1, 1].set_title("ZFP-perturbed")
            axs[1, 1].set_xticks([])
            axs[1, 1].set_yticks([])

            if pyblaz_perturbations is not None:
                axs[2, 0].imshow(
                    data_type[1][pyblaz_worst_index].reshape(size, size),
                    cmap=sequential_colormap if "y" in data_type[0] else diverging_colormap,
                    vmin=0 if "y" in data_type[0] else vmin,
                    vmax=vmax,
                )
                axs[2, 0].set_title("original")
                axs[2, 0].set_xticks([])
                axs[2, 0].set_yticks([])
                axs[2, 1].imshow(
                    data_type[1][pyblaz_worst_index].reshape(size, size)
                    + closest_pyblaz_perturbation[pyblaz_worst_index].reshape(size, size),
                    cmap=sequential_colormap if "y" in data_type[0] else diverging_colormap,
                    vmin=0 if "y" in data_type[0] else vmin,
                    vmax=vmax,
                )
                axs[2, 1].set_title(make_pyblaz_label(closest_pyblaz_perturbation_name))
                axs[2, 1].set_xticks([])
                axs[2, 1].set_yticks([])

            # [left, bottom, width, height] in figure units
            colorbar_ax1 = fig.add_axes([0.65, 0.03, 0.01, 0.9])
            fig.colorbar(cax1, cax=colorbar_ax1)

            cax2 = axs[0, 2].imshow(
                sz_perturbed_data[sz_worst_index].reshape(size, size)
                - data_type[1][sz_worst_index].reshape(size, size),
                cmap=diverging_colormap,
            )
            axs[0, 2].set_title("difference")
            axs[0, 2].set_xticks([])
            axs[0, 2].set_yticks([])

            axs[1, 2].imshow(
                zfp_perturbed_data[zfp_worst_index].reshape(size, size)
                - data_type[1][zfp_worst_index].reshape(size, size),
                cmap=diverging_colormap,
            )
            axs[1, 2].set_title("difference")
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])

            if pyblaz_perturbations is not None:
                axs[2, 2].imshow(
                    closest_pyblaz_perturbation[pyblaz_worst_index].reshape(size, size),
                    cmap=diverging_colormap,
                )
                axs[2, 2].set_title("difference")
                axs[2, 2].set_xticks([])
                axs[2, 2].set_yticks([])

            colorbar_ax2 = fig.add_axes([0.95, 0.03, 0.01, 0.9])
            fig.colorbar(cax2, cax=colorbar_ax2)

            plt.suptitle(f"Biggest perturbations, {pretty_subset_names[data_type[0]]}, {std_multiplier}$\sigma$")
            plt.savefig(combined_save_folder / f"{data_type[0]}_worst_{std_multiplier}.{args.figure_format}", dpi=300)
            plt.close()

            plot_histogram(
                perturbations,
                labels,
                bins,
                abs_range,
                f"Perturbation distribution, {pretty_subset_names[data_type[0]]}, {std_multiplier}$\sigma$",
                combined_save_folder / f"{data_type[0]}_hist_{std_multiplier}.{args.figure_format}",
            )

            sz_magnitudes, sz_phases = get_magnitudes_and_phases(sz_perturbed_data, size, device)
            zfp_magnitudes, zfp_phases = get_magnitudes_and_phases(zfp_perturbed_data, size, device)
            magnitudes_arrays = [sz_magnitudes, zfp_magnitudes]
            phases_arrays = [sz_phases, zfp_phases]
            names = ["SZ", "ZFP"]
            if pyblaz_perturbations is not None:
                closest_pyblaz_magnitudes, closest_pyblaz_phases = get_magnitudes_and_phases(
                    closest_pyblaz_perturbation, size, device
                )
                magnitudes_arrays.append(closest_pyblaz_magnitudes)
                phases_arrays.append(closest_pyblaz_phases)
                names.append(make_pyblaz_label(closest_pyblaz_perturbation_name))

            plot_fft(
                magnitudes_arrays,
                phases_arrays,
                names,
                subplot_size,
                f"Perturbation frequencies, {pretty_subset_names[data_type[0]]}, {std_multiplier}$\sigma$",
                fft_save_folder / f"{data_type[0]}_fft_{std_multiplier}.{args.figure_format}",
            )
            progress_bar.update()

    progress_bar.close()


def read_binary_and_reshape(file_path, size):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, size**2)


def get_magnitudes_and_phases(data, size, device):
    data_tensor = torch.tensor(data).reshape(-1, size, size).to(device)
    fft_result = torch.fft.fft2(data_tensor)
    magnitudes = fft_result.abs().mean(dim=0).cpu().numpy()
    phases = fft_result.angle().mean(dim=0).cpu().numpy()
    return magnitudes, phases


def plot_histogram(perturbations, labels, bins, abs_range, title, save_path):
    for perturbation, label in zip(perturbations, labels):
        plt.hist(perturbation.flatten(), bins=bins, range=(-abs_range, abs_range), density=True, label=label, alpha=0.5)
    plt.xlabel("perturbation")
    plt.ylabel("density")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_fft(magnitudes_arrays, phases_arrays, names, subplot_size, title, save_path):
    num_summaries = len(names)

    fig, axs = plt.subplots(
        2, num_summaries, figsize=(subplot_size * num_summaries, subplot_size * 2), constrained_layout=True
    )

    for i, (magnitudes_array, phases_array, name) in enumerate(zip(magnitudes_arrays, phases_arrays, names)):
        # Plot magnitude
        mag_ax = axs[0, i]
        cax_mag = mag_ax.imshow(magnitudes_array, norm=colors.LogNorm(vmin=magnitudes_array.min() + 1), aspect="equal")
        mag_ax.set_title(name)
        mag_ax.set_xticks([])
        mag_ax.set_yticks([])

        # Plot phase
        phase_ax = axs[1, i]
        cax_phase = phase_ax.imshow(phases_array, cmap="twilight", aspect="equal")

        phase_ax.set_xticks([])
        phase_ax.set_yticks([])

        # Adjust colorbar for magnitude and phase if needed
        if i == num_summaries - 1:  # Add colorbar to the last pair for both magnitude and phase
            fig.colorbar(cax_mag, ax=mag_ax, location="right", fraction=0.046, pad=0.04)
            fig.colorbar(cax_phase, ax=phase_ax, location="right", fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.savefig(save_path, dpi=300)
    plt.close()


def gather_pyblaz_perturbations(args, y_train, y_test, x_train_branch, x_test_branch):
    perturbations = {}
    progress_bar = tqdm.tqdm(total=2 * 1 * 2 * 4, desc="Gathering PyBlaz perturbations")
    for block_size in (4, 8):
        for float_type in ("float32",):
            for index_dtype in ("int8", "int16"):
                pyblaz_perturbed_folder = (
                    pathlib.Path(args.pyblaz_perturbed_folder)
                    / f"r{args.resolution}"
                    / f"bs{block_size}_{float_type}_{index_dtype}"
                )
                perturbed_x_train_branch = np.load(pyblaz_perturbed_folder / "x_train_branch.npy")
                progress_bar.update()

                perturbed_x_test_branch = np.load(pyblaz_perturbed_folder / "x_test_branch.npy")
                progress_bar.update()

                perturbed_y_train = np.load(pyblaz_perturbed_folder / "y_train.npy")
                progress_bar.update()

                perturbed_y_test = np.load(pyblaz_perturbed_folder / "y_test.npy")

                x_train_branch_perturbation = perturbed_x_train_branch - x_train_branch
                x_test_branch_perturbation = perturbed_x_test_branch - x_test_branch
                y_train_perturbation = perturbed_y_train - y_train
                y_test_perturbation = perturbed_y_test - y_test

                perturbations[f"bs{block_size}_{float_type}_{index_dtype}"] = {
                    "x_train_branch": x_train_branch_perturbation,
                    "x_test_branch": x_test_branch_perturbation,
                    "y_train": y_train_perturbation,
                    "y_test": y_test_perturbation,
                }
                progress_bar.update()
    progress_bar.close()
    return perturbations


def make_pyblaz_label(ugly_string):
    return f"PyBlaz {ugly_string.replace('bs', 'BS').replace('_', ' ')}"


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
