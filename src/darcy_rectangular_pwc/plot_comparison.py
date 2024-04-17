import argparse
import pathlib
import scipy.io
import scipy.stats

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
    parser.add_argument("--best", type=int, default=None)
    parser.add_argument("--worst", type=int, default=None)
    parser.add_argument("--criterion", type=str, default=None, choices=[None, "l2", "linf"])
    parser.add_argument(
        "--include-clean", action="store_true", help="Plot clean with top schemes even if it's not in top."
    )
    parser.add_argument(
        "--pyblaz-floats",
        action="store_true",
        help="Differentiate between float types in PyBlaz. Otherwise, removes float32 from labels.",
    )
    args = parser.parse_args()

    results_folder = pathlib.Path(args.results_folder)
    figures_folder = pathlib.Path(args.figures_folder) / "error_comparison"
    figures_folder.mkdir(parents=True, exist_ok=True)

    must_include_words = (f"r{args.resolution}",) + (tuple(args.must_include) if args.must_include else tuple())
    plot_compression_degrees_experiment(
        args,
        must_include_words,
        tuple(args.may_include) if args.may_include else tuple(),
        tuple(args.exclude) if args.exclude else tuple(),
        results_folder,
        figures_folder / args.plot_prefix,
    )


def plot_compression_degrees_experiment(
    args, must_include_words, may_include_words, exclude_words, results_folder, plot_file_prefix
):
    hyperparameter_subfolders = [
        folder
        for folder in results_folder.iterdir()
        if folder.is_dir()
        and (
            all(word in folder.name for word in must_include_words)
            or any(word in folder.name for word in may_include_words)
        )
        and not any(word in folder.name for word in exclude_words)
        or (args.include_clean and f"r{args.resolution}_clean" in folder.name)
    ]

    sort_criteria = [
        ["clean", "all", "half"],
        ["train", "test", "both"],
        ["bs4", "bs8"],
        ["int8", "int16"],
        ["0.1", "0.01", "0.001", "0.0001"],
    ]
    hyperparameter_subfolders.sort(key=lambda folder: sort_key(folder.name, sort_criteria))

    if args.include_clean:
        # Should be the first folder.
        clean_mean_l2_errors = []
        clean_mean_linf_errors = []
        for trial_folder in sorted(pathlib.Path(hyperparameter_subfolders[0]).iterdir()):
            if (errors_file := trial_folder / "errors.npy").exists():
                element_wise_errors = torch.tensor(np.load(errors_file))
                clean_mean_l2_errors.append(element_wise_errors.norm(2, (1, 2)).mean().item())
                clean_mean_linf_errors.append(element_wise_errors.norm(torch.inf, (1, 2)).mean().item())

    else:
        print("\nFound\n" + "\n".join(str(folder.name) for folder in hyperparameter_subfolders))

    mean_mean_l2_errors = []
    std_mean_l2_errors = []

    mean_mean_linf_errors = []
    std_mean_linf_errors = []

    for subfolder in (progress_bar := tqdm.tqdm(hyperparameter_subfolders)):
        progress_bar.set_description(f"{subfolder.name}")

        mean_l2_errors = []
        mean_linf_errors = []

        for trial_folder in sorted(pathlib.Path(subfolder).iterdir()):
            if (errors_file := trial_folder / "errors.npy").exists():
                element_wise_errors = torch.tensor(np.load(errors_file))
                l2_errors = element_wise_errors.norm(2, (1, 2))
                mean_l2_errors.append(l2_errors.mean().item())

                linf_errors = element_wise_errors.norm(torch.inf, (1, 2))
                mean_linf_errors.append(linf_errors.mean().item())

        mean_mean_l2_errors.append(torch.tensor(mean_l2_errors).mean().item())
        std_mean_l2_errors.append(torch.tensor(mean_l2_errors).std(correction=1).item())
        mean_mean_linf_errors.append(torch.tensor(mean_linf_errors).mean().item())
        std_mean_linf_errors.append(torch.tensor(mean_linf_errors).std(correction=1).item())

        if args.include_clean:
            progress_bar.write(
                f"clean vs {subfolder.name} p-values "
                f"L2: {scipy.stats.ttest_ind(clean_mean_l2_errors, mean_l2_errors).pvalue}, "
                f"Linf: {scipy.stats.ttest_ind(clean_mean_linf_errors, mean_linf_errors).pvalue}"
            )

    if selective := bool(args.best or args.worst):
        if args.criterion == "l2":
            indices = np.argsort(mean_mean_l2_errors)[-args.worst if args.worst else None : args.best]
        elif args.criterion == "linf":
            indices = np.argsort(mean_mean_linf_errors)[-args.worst if args.worst else None : args.best]
        plot_mean_mean_l2_errors = [mean_mean_l2_errors[i] for i in indices]
        plot_std_mean_l2_errors = [std_mean_l2_errors[i] for i in indices]
        plot_mean_mean_linf_errors = [mean_mean_linf_errors[i] for i in indices]
        plot_std_mean_linf_errors = [std_mean_linf_errors[i] for i in indices]
        plot_hyperparameter_subfolders = [hyperparameter_subfolders[i] for i in indices]
    else:
        plot_mean_mean_l2_errors = mean_mean_l2_errors
        plot_std_mean_l2_errors = std_mean_l2_errors
        plot_mean_mean_linf_errors = mean_mean_linf_errors
        plot_std_mean_linf_errors = std_mean_linf_errors
        plot_hyperparameter_subfolders = hyperparameter_subfolders
    if args.include_clean and all("clean" not in folder.name for folder in plot_hyperparameter_subfolders):
        for i, folder in enumerate(hyperparameter_subfolders):
            if "clean" in folder.name:
                plot_mean_mean_l2_errors.insert(0, mean_mean_l2_errors[i])
                plot_std_mean_l2_errors.insert(0, std_mean_l2_errors[i])
                plot_mean_mean_linf_errors.insert(0, mean_mean_linf_errors[i])
                plot_std_mean_linf_errors.insert(0, std_mean_linf_errors[i])
                plot_hyperparameter_subfolders.insert(0, folder)
                break

    title = (
        f"{('Best ' + str(args.best) + ' ') * bool(args.best)}{('Worst ' + str(args.worst) + ' ') * bool(args.worst)}"
        f"{_pretty_criterion_name(args.criterion) if bool(args.criterion) else ''}error {'per ' * (not selective)}perturbation scheme{'s' * selective}"
    )

    # If it's all the same compressor, remove the name from labels and put it in the title
    all_same_compressor = False
    if all("pyblaz" in folder.name or "clean" in folder.name for folder in plot_hyperparameter_subfolders):
        all_same_compressor = True
        title = "PyBlaz " + title
    elif all("zfp" in folder.name or "clean" in folder.name for folder in plot_hyperparameter_subfolders):
        all_same_compressor = True
        title = "ZFP " + title
    elif all("sz" in folder.name or "clean" in folder.name for folder in plot_hyperparameter_subfolders):
        all_same_compressor = True
        title = "SZ " + title

    schemes = [_pretty_label_name(subfolder.name, all_same_compressor) for subfolder in plot_hyperparameter_subfolders]
    # We usually only show float32 in PyBlaz. Remove it from the labels to avoid clutter.
    if not args.pyblaz_floats:
        schemes = [scheme.replace(" float32", "") for scheme in schemes]

    print("\nPlotting\n" + "\n".join(str(folder.name) for folder in plot_hyperparameter_subfolders))
    horizontal_positions = np.arange(len(schemes))
    offset = 0.2

    fig, ax1 = plt.subplots()
    plt.title(title)

    fig.set_size_inches(len(schemes) / 1.5, 4)
    ax1.errorbar(
        horizontal_positions - offset / 2,
        plot_mean_mean_l2_errors,
        yerr=plot_std_mean_l2_errors,
        fmt="o",
        color="tab:blue",
    )
    ax1.set_xlabel("perturbation scheme")
    ax1.set_ylabel("$L_2$ error", color="tab:blue")
    ax1.set_xticks(horizontal_positions)
    ax1.set_xticklabels(
        schemes,
        rotation=-30,
        # ha="center",
        ha="left",
    )

    ax2 = ax1.twinx()
    ax2.errorbar(
        horizontal_positions + offset / 2,
        plot_mean_mean_linf_errors,
        yerr=plot_std_mean_linf_errors,
        fmt="o",
        color="tab:orange",
    )
    ax2.set_ylabel("$L_\\infty$ error", color="tab:orange")

    # Put horizontal lines at the clean error.
    for i, scheme in enumerate(schemes):
        if "clean" in scheme:
            ax1.axhline(plot_mean_mean_l2_errors[i], color="tab:blue", linestyle="--", alpha=0.25)
            ax2.axhline(plot_mean_mean_linf_errors[i], color="tab:orange", linestyle="--", alpha=0.25)
            break

    fig.tight_layout()

    plt.savefig(
        plot_file_prefix / f"{('all_' + '-'.join(must_include_words) + '_') * bool(must_include_words)}"
        f"{('any_' + '-'.join(may_include_words) + '_') * bool(may_include_words)}"
        f"{('no_' + '-'.join(exclude_words) + '_') * bool(exclude_words)}"
        "mean_errors"
        f"{('_best_' + str(args.best)) * bool(args.best)}"
        f"{('_worst_' + str(args.worst)) * bool(args.worst)}"
        f"{('_' + args.criterion) if selective else ''}"
        f".{args.figure_format}",
        dpi=300,
    )
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


def _pretty_label_name(subfolder_name, all_same_compressor=False):
    return (
        subfolder_name[3:]
        .replace("replace_", "_")
        .replace("pyblaz_", "PyBlaz_" if not all_same_compressor else "")
        .replace("zfp_", "ZFP_" if not all_same_compressor else "")
        .replace("sz_", "SZ_" if not all_same_compressor else "")
        .replace("bs", "BS")
        .replace("1_", "1$\sigma$_")
        .replace("_", " ")
    )


def _pretty_criterion_name(criterion):
    return {"l2": "$L_2$", "linf": "$L_\\infty$"}[criterion]


def sort_key(strings: list, criteria: list) -> tuple:
    return tuple(
        next((criterion.index(c) for c in criterion if c in strings), len(criterion)) for criterion in criteria
    )


if __name__ == "__main__":
    main()
