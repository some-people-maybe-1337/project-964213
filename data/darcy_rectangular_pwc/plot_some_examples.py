import pathlib
import argparse

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--save-folder", type=str, default="some_examples")
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--figure-format", type=str, default="pdf")
    args = parser.parse_args()

    n_train = 1024  # up to 1024
    n_test = 1024  # up to 1024

    data_folder = pathlib.Path("/datasets") / "lulu_cmame2022" / "Darcy_rectangular_PWC"
    save_folder = pathlib.Path(args.save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)

    x_train, y_train = get_data(data_folder / "piececonst_r421_N1024_smooth1.mat", n_train, resolution=args.resolution)
    # x_test, y_test = get_data(data_folder / "piececonst_r421_N1024_smooth2.mat", n_test, resolution=args.resolution)

    size = int(((421 - 1) / args.resolution) + 1)

    figure, ax = plt.subplots()
    figure.set_size_inches(6, 3)

    grid = x_train[1].reshape(size, size, 2)
    grid = np.concatenate([grid, np.zeros((size, size, 1))], axis=-1)

    ax.imshow(grid)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("$y$")

    norm = mcolors.Normalize(vmin=grid[:, :, :2].min(), vmax=grid[:, :, :2].max())

    # Create colorbar for the Red channel
    colors_red = [(0, 0, 0), (1, 0, 0)]
    cmap_red = mcolors.LinearSegmentedColormap.from_list("BlackToRed", colors_red)
    sm_red = plt.cm.ScalarMappable(cmap=cmap_red, norm=norm)
    sm_red.set_array([])
    cb_ax_red = figure.add_axes((0.73, 0.1, 0.02, 0.775))
    cb_red = figure.colorbar(sm_red, cax=cb_ax_red)
    cb_red.set_label("permeability")

    # Create colorbar for the Green channel
    colors_green = [(0, 0, 0), (0, 1, 0)]
    cmap_green = mcolors.LinearSegmentedColormap.from_list("BlackToGreen", colors_green)
    sm_green = plt.cm.ScalarMappable(cmap=cmap_green, norm=norm)
    sm_green.set_array([])
    cb_ax_green = figure.add_axes((0.85, 0.1, 0.02, 0.775))
    cb_green = figure.colorbar(sm_green, cax=cb_ax_green)
    cb_green.set_label("pressure")
    plt.savefig(save_folder / f"grid.{args.figure_format}", dpi=300, bbox_inches="tight")
    plt.close(figure)

    for i in range(args.n_examples):
        idx = np.random.randint(0, n_train)
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(7.5, 3)
        im = axs[0].imshow(x_train[0][idx].reshape(size, size), cmap="seismic")
        axs[0].set_title("$u(x)$")
        axs[0].axis("off")
        fig.colorbar(im, ax=axs[0])

        im2 = axs[1].imshow(y_train[idx].reshape(size, size))
        axs[1].set_title("$v(y)$")
        axs[1].axis("off")
        fig.colorbar(im2, ax=axs[1])

        plt.tight_layout()
        plt.savefig(save_folder / f"{i}.{args.figure_format}", dpi=300)
        plt.close(fig)


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
