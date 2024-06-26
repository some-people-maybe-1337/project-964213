import argparse

import deepxde as dde
import numpy as np

# import matplotlib.pyplot as plt
import tqdm

from advection import Advection, Advection_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ic", type=int, default=2, help="Initial condition type")
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--space-size", type=int, default=128)
    parser.add_argument("--time-size", type=int, default=128)
    parser.add_argument("--save-name", type=str, default="advection_ic2.npz")
    args = parser.parse_args()

    main_IC2(args.n_examples, args.space_size, args.time_size, args.save_name)


def main_IC1():
    ndata = 1000
    nx = 40
    nt = 40

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    x = geomtime.uniform_points(nx * nt, boundary=True)

    u = []
    rng = np.random.default_rng()
    for i in range(ndata):
        height = rng.random() + 1  # [1, 2]
        width = 0.3 * rng.random() + 0.3  # [0.3, 0.6]
        x0 = 0.4 * rng.random() + 0.3  # [0.3, 0.7]
        pde = Advection(height, width, 1, x0, 0, 1)
        u.append(pde.solve(x).reshape(nt, nx))
    u = np.array(u)
    x, t = x[:, 0].reshape(nt, nx), x[:, 1].reshape(nt, nx)
    np.savez_compressed("train.npz", x=x, t=t, u=u)

    # for i in range(3):
    #     plt.figure()
    #     plt.imshow(u[i])
    #     plt.colorbar()
    # plt.show()


def main_IC2(n_examples=1000, space_size=40, time_size=40, save_name="advection_ic2.npz"):
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    x = geomtime.uniform_points(space_size * time_size, boundary=True)

    u = []
    rng = np.random.default_rng()
    for i in tqdm.tqdm(range(n_examples), desc="Generating data"):
        c1 = 0.1 * rng.random() + 0.2  # [0.2, 0.3]
        w = 0.1 * rng.random() + 0.1  # [0.1, 0.2]
        h1 = 1.5 * rng.random() + 0.5  # [0.5, 2]
        c2 = 0.1 * rng.random() + 0.7  # [0.7, 0.8]
        a = 5 * rng.random() + 5  # 1 / [0.1, 0.2] = [5, 10]
        h2 = 1.5 * rng.random() + 0.5  # [0.5, 2]
        pde = Advection_v2(0, 1, c1, w, h1, c2, a, h2)
        u.append(pde.solve(x).reshape(time_size, space_size))
    u = np.array(u)
    x, t = x[:, 0].reshape(time_size, space_size), x[:, 1].reshape(time_size, space_size)
    np.savez_compressed(save_name, x=x, t=t, u=u)

    # for i in range(3):
    #     plt.figure()
    #     plt.imshow(u[i])
    #     plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    main()
