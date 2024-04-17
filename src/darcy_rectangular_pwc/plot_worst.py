import argparse
import pathlib
import scipy.io

import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-folder", type=str, default="results")
    parser.add_argument("--figures-folder", type=str, default="figures")
    parser.add_argument("--resolution", "-r", type=int, default=1)
    parser.add_argument("--plot-prefix", type=str, default="")
    parser.add_argument("--figure-format", type=str, default="pdf")
    parser.add_argument("--must-include", nargs="+", default=None)
    parser.add_argument("--may-include", nargs="+", default=None)
    parser.add_argument("--exclude", nargs="+", default=None)
    args = parser.parse_args()

    results_folder = pathlib.Path(args.results_folder)
    figures_folder = pathlib.Path(args.figures_folder) / "worst"
    figures_folder.mkdir(parents=True, exist_ok=True)

    must_include_words = (f"r{args.resolution}",) + (tuple(args.must_include) if args.must_include else tuple())
    plot_worst_examples(
        args,
        must_include_words,
        (tuple(args.may_include) if args.may_include else tuple()),
        (tuple(args.exclude) if args.exclude else tuple()),
        results_folder,
        figures_folder / args.plot_prefix,
    )


def plot_worst_examples(args, must_include_words, may_include_words, exclude_words, results_folder, plot_file_prefix):
    # Need to load data to get relative errors.
    # n_test = 256
    # data_folder = pathlib.Path("/datasets") / "Darcy_rectangular_PWC"
    # _, y_test = get_data(data_folder / "piececonst_r421_N1024_smooth2.mat", n_test, resolution=args.resolution)
    # grid_size = int(((421 - 1) / args.resolution) + 1)
    # y_test = torch.tensor(y_test).reshape(-1, grid_size, grid_size)

    hyperparameter_subfolders = sorted(
        [
            folder
            for folder in results_folder.iterdir()
            if folder.is_dir()
            and (
                all(word in folder.name for word in must_include_words)
                or any(word in folder.name for word in may_include_words)
            )
            and not any(word in folder.name for word in exclude_words)
        ]
    )
    # sort again by clean, train, test, both
    hyperparameter_subfolders = sorted(
        hyperparameter_subfolders,
        key=lambda folder: (
            0 if "clean" in folder.name else 1 if "train" in folder.name else 2 if "test" in folder.name else 3
        ),
    )

    print("Found\n" + "\n".join(str(folder.name) for folder in hyperparameter_subfolders))
    mean_mean_l2_errors = []
    std_mean_l2_errors = []

    mean_mean_linf_errors = []
    std_mean_linf_errors = []

    # mean_mean_l2_relative_errors = []
    # std_mean_l2_relative_errors = []

    for subfolder in (progress_bar := tqdm.tqdm(hyperparameter_subfolders)):
        progress_bar.set_description(f"{subfolder.name}")

        worst_element_wise_errors_in_l2 = []
        mean_l2_errors = []
        worst_l2_errors = []
        worst_l2_errors_indices = []

        worst_element_wise_errors_in_linf = []
        mean_linf_errors = []
        worst_linf_errors = []
        worst_linf_errors_indices = []

        # mean_l2_relative_errors = []

        for trial_folder in sorted(pathlib.Path(subfolder).iterdir()):
            if (errors_file := trial_folder / "errors.npy").exists():
                element_wise_errors = torch.tensor(np.load(errors_file))
                l2_errors = element_wise_errors.norm(2, (1, 2))
                worst_l2, worst_l2_index = l2_errors.max(0)
                worst_l2_errors.append(worst_l2.item())
                worst_l2_errors_indices.append(worst_l2_index.item())
                worst_element_wise_errors_in_l2.append(element_wise_errors[worst_l2_index])
                mean_l2_errors.append(l2_errors.mean().item())

                linf_errors = element_wise_errors.norm(torch.inf, (1, 2))
                mean_linf_errors.append(linf_errors.mean().item())
                worst_linf, worst_linf_index = linf_errors.max(0)
                worst_linf_errors.append(worst_linf.item())
                worst_linf_errors_indices.append(worst_linf_index.item())
                worst_element_wise_errors_in_linf.append(element_wise_errors[worst_linf_index])

                # l2_of_validation_targets = y_test.norm(2, (1, 2))
                # l2_relative_errors = l2_errors / l2_of_validation_targets
                # mean_l2_relative_errors.append(l2_relative_errors.mean().item())

        mean_mean_l2_errors.append(torch.tensor(mean_l2_errors).mean().item())
        std_mean_l2_errors.append(torch.tensor(mean_l2_errors).std(correction=1).item())
        mean_mean_linf_errors.append(torch.tensor(worst_linf_errors).mean().item())
        std_mean_linf_errors.append(torch.tensor(worst_linf_errors).std(correction=1).item())
        # mean_mean_l2_relative_errors.append(torch.tensor(mean_l2_relative_errors).mean().item())
        # std_mean_l2_relative_errors.append(torch.tensor(mean_l2_relative_errors).std(correction=1).item())

        subplot_size = 3
        fig, axs = plt.subplots(
            1,
            len(worst_l2_errors),
            figsize=(subplot_size * len(worst_l2_errors), subplot_size),
            constrained_layout=True,
        )
        vmax = max(array.norm(torch.inf) for array in worst_element_wise_errors_in_l2)
        vmin = -vmax
        for trial_number, (ax, array) in enumerate(zip(axs, worst_element_wise_errors_in_l2)):
            cax = ax.imshow(array.cpu().numpy(), cmap="seismic", aspect="equal", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"example {worst_l2_errors_indices[trial_number]}, $L_2$ {worst_l2_errors[trial_number]:.2f}, $L_\\infty$ {array.norm(torch.inf).item():.2f}"
            )
        fig.colorbar(cax, ax=axs.ravel().tolist(), location="right")
        plt.savefig(plot_file_prefix / f"{subfolder.name}_l2.{args.figure_format}")
        plt.close()

        fig, axs = plt.subplots(
            1,
            len(worst_element_wise_errors_in_linf),
            figsize=(subplot_size * len(worst_element_wise_errors_in_linf), subplot_size),
            constrained_layout=True,
        )
        vmax = max(array.norm(torch.inf) for array in worst_element_wise_errors_in_linf)
        vmin = -vmax
        for trial_number, (ax, array) in enumerate(zip(axs, worst_element_wise_errors_in_linf)):
            cax = ax.imshow(array.cpu().numpy(), cmap="seismic", aspect="equal", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"example {worst_linf_errors_indices[trial_number]}, $L_2$ {array.norm(2).item():.2f} $L_\\infty$ {worst_linf_errors[trial_number]:.2f}"
            )
        fig.colorbar(cax, ax=axs.ravel().tolist(), location="right")
        plt.savefig(plot_file_prefix / f"{subfolder.name}_linf.{args.figure_format}")
        plt.close()


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


def _pretty_name(subfolder_name):
    return subfolder_name[3:].replace("_", " ").replace("pyblaz", "PyBlaz").replace("zfp", "ZFP")


if __name__ == "__main__":
    main()
