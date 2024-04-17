import argparse
import pathlib
import deepxde as dde
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tqdm

from advection import Advection, Advection_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic", type=int, default=2, help="Initial condition type")
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--space-size", type=int, default=128)
    parser.add_argument("--time-size", type=int, default=128)
    parser.add_argument("--figures-folder", type=str, default="some_examples")
    parser.add_argument("--figure-format", type=str, default="pdf")
    args = parser.parse_args()

    main_IC2(args)


def main_IC2(args):
    figures_folder = pathlib.Path(args.figures_folder)
    figures_folder.mkdir(parents=True, exist_ok=True)

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    x = geomtime.uniform_points(args.space_size * args.time_size, boundary=True)

    u = []
    rng = np.random.default_rng()
    for i in tqdm.tqdm(range(args.n_examples), desc="Generating data"):
        c1 = 0.1 * rng.random() + 0.2  # [0.2, 0.3]
        w = 0.1 * rng.random() + 0.1  # [0.1, 0.2]
        h1 = 1.5 * rng.random() + 0.5  # [0.5, 2]
        c2 = 0.1 * rng.random() + 0.7  # [0.7, 0.8]
        a = 5 * rng.random() + 5  # 1 / [0.1, 0.2] = [5, 10]
        h2 = 1.5 * rng.random() + 0.5  # [0.5, 2]
        pde = Advection_v2(0, 1, c1, w, h1, c2, a, h2)
        u.append(pde.solve(x).reshape(args.time_size, args.space_size))
    u = np.array(u)
    x, t = x[:, 0].reshape(args.time_size, args.space_size), x[:, 1].reshape(args.time_size, args.space_size)

    u0 = u[:, 0, :]
    xt = np.vstack((np.ravel(x), np.ravel(t))).T
    u = u.reshape(-1, args.time_size * args.space_size)

    # Names matter.
    u_of_x = u0
    y = xt.reshape(args.time_size, args.space_size, 2)
    v_of_y = u.reshape(args.n_examples, args.time_size, args.space_size)

    # Pretend y is a 3-channel image.
    y = np.concatenate((y, np.zeros((128, 128, 1))), axis=-1)

    figure, ax = plt.subplots()
    figure.set_size_inches(6, 3)

    ax.imshow(y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("$y$")

    norm = mcolors.Normalize(vmin=y[:, :, :2].min(), vmax=y[:, :, :2].max())

    # Create colorbar for the Red channel
    colors_red = [(0, 0, 0), (1, 0, 0)]
    cmap_red = mcolors.LinearSegmentedColormap.from_list("BlackToRed", colors_red)
    sm_red = plt.cm.ScalarMappable(cmap=cmap_red, norm=norm)
    sm_red.set_array([])
    cb_ax_red = figure.add_axes((0.73, 0.1, 0.02, 0.775))
    cb_red = figure.colorbar(sm_red, cax=cb_ax_red)
    cb_red.set_label("time")

    # Create colorbar for the Green channel
    colors_green = [(0, 0, 0), (0, 1, 0)]
    cmap_green = mcolors.LinearSegmentedColormap.from_list("BlackToGreen", colors_green)
    sm_green = plt.cm.ScalarMappable(cmap=cmap_green, norm=norm)
    sm_green.set_array([])
    cb_ax_green = figure.add_axes((0.85, 0.1, 0.02, 0.775))
    cb_green = figure.colorbar(sm_green, cax=cb_ax_green)
    cb_green.set_label("space")
    plt.savefig(figures_folder / f"y.{args.figure_format}", dpi=300, bbox_inches="tight")
    plt.close(figure)

    for i in range(args.n_examples):
        figure = plt.figure()
        figure.set_size_inches(4, 3)
        plt.imshow(v_of_y[i], cmap="viridis", aspect="equal")
        plt.xticks([])
        plt.yticks([])
        plt.title(f"$u(x)$ and $v(y)$")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(figures_folder / f"{i}.{args.figure_format}", dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
