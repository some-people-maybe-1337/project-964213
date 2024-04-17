import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import deepxde as dde
from deepxde.backend import tf

from numpy import savez_compressed
import zfpy


def get_data(filename):
    nx = 128
    nt = 128
    data = np.load(filename)
    x = data["x"].astype(np.float32)
    t = data["t"].astype(np.float32)
    u = data["u"].astype(np.float32)  # N x nt x nx

    u0 = u[:, 0, :]  # N x nx
    xt = np.vstack((np.ravel(x), np.ravel(t))).T
    u = u.reshape(-1, nt * nx)
    return (u0, xt), u


def save_data_zfp(x_train, y_train, x_test, y_test, multiplier):
    x_train_0 = zfpy.decompress_numpy(zfpy.compress_numpy(x_train[0], tolerance=multiplier * np.std(x_train[0])))
    x_train_1 = zfpy.decompress_numpy(zfpy.compress_numpy(x_train[1], tolerance=multiplier * np.std(x_train[1])))
    y_train = zfpy.decompress_numpy(zfpy.compress_numpy(y_train, tolerance=multiplier * np.std(y_train)))
    x_test_0 = zfpy.decompress_numpy(zfpy.compress_numpy(x_test[0], tolerance=multiplier * np.std(x_test[0])))
    x_test_1 = zfpy.decompress_numpy(zfpy.compress_numpy(x_test[1], tolerance=multiplier * np.std(x_test[1])))
    y_test = zfpy.decompress_numpy(zfpy.compress_numpy(y_test, tolerance=multiplier * np.std(y_test)))
    x_train_0.tofile("x_train_0_128X128_size1000.dat")
    x_train_1.tofile("x_train_1_128X128_size1000.dat")
    y_train.tofile("y_train_128X128_size1000.dat")
    x_test_0.tofile("x_test_0_128X128_size1000.dat")
    x_test_1.tofile("x_test_1_128X128_size1000.dat")
    y_test.tofile("y_test_128X128_size1000.dat")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--set", type=str, choices=(None, "train", "test", "both"), default=None)
    parser.add_argument("--subset_train", type=str, choices=(None, "x", "y", "both"), default=None)
    parser.add_argument("--subset_test", type=str, choices=(None, "x", "y", "both"), default=None)
    parser.add_argument("--compressor", type=str, choices=(None, "zfp"), default=None)
    parser.add_argument("--augment", type=str, choices=(None, "full", "half"), default=None)
    # parser.add_argument("--augment", action="store_true", default=False)
    # parser.add_argument("--input-folder", type=str, default="original_dat/decompressed_SZ3/")
    parser.add_argument("--multiplier", type=float, default=0.0001)
    # parser.add_argument("--results-folder", type=str, default="debug")
    parser.add_argument("--epochs", type=int, default=250000)
    args = parser.parse_args()
    input_folder = "original_dat_128X128_size1000/ZFP/"
    # results_folder = pathlib.Path(args.results_folder)
    # input_folder = pathlib.Path(args.input_folder)

    nt = 128
    nx = 128
    x_train, y_train = get_data("original_dat_128X128_size1000/advection_ic2_train_128X128_size1000.npz")
    x_test, y_test = get_data("original_dat_128X128_size1000/advection_ic2_test_128X128_size1000.npz")
    # save_data_zfp(x_train, y_train, x_test, y_test, args.multiplier)
    # Clean

    if args.compressor == "zfp":
        # Perturbed train (full) “All our data came compressed”
        if args.augment == None and (args.set == "train" and args.subset_train == "both"):
            name = "case1"

            x_train_0 = np.fromfile(
                input_folder + "eb" + str(args.multiplier) + "/decompressed_dat/x_train_0_128X128_size1000.dat",
                dtype=np.float32,
            )
            x_train = (x_train_0.reshape(1000, 128), x_train[1])
            y_train = np.fromfile(
                input_folder + "eb" + str(args.multiplier) + "/decompressed_dat/y_train_128X128_size1000.dat",
                dtype=np.float32,
            )
            y_train = y_train.reshape(1000, 16384)

        # Perturbed test (full) (except test y) “We’re deploying using compressed data”
        if args.augment == None and (args.set == "test" and args.subset_test == "x"):
            name = "case2"
            x_test_0 = np.fromfile(
                input_folder + "eb" + str(args.multiplier) + "/decompressed_dat/x_test_0_128X128_size1000.dat",
                dtype=np.float32,
            )
            x_test = (x_test_0.reshape(1000, 128), x_test[1])

        # Perturbed both (full) (except test y) “We’ve trained on compressed data, deploying on compressed”
        if args.augment == None and (args.set == "both" and (args.subset_train == "both" and args.subset_test == "x")):
            name = "case3"
            x_train_0 = np.fromfile(
                input_folder + "eb" + str(args.multiplier) + "/decompressed_dat/x_train_0_128X128_size1000.dat",
                dtype=np.float32,
            )
            x_train = (x_train_0.reshape(1000, 128), x_train[1])
            y_train = np.fromfile(
                input_folder + "eb" + str(args.multiplier) + "/decompressed_dat/y_train_128X128_size1000.dat",
                dtype=np.float32,
            )
            y_train = y_train.reshape(1000, 16384)

            x_test_0 = np.fromfile(
                input_folder + "eb" + str(args.multiplier) + "/decompressed_dat/x_test_0_128X128_size1000.dat",
                dtype=np.float32,
            )
            x_test = (x_test_0.reshape(1000, 128), x_test[1])

        # Augmented train (full clean + full perturbed) => double data size === robustness
        if args.augment == "full" and (args.set == "train" and args.subset_train == "both"):
            name = "case4"
            x_train_0_noisy = np.fromfile(
                input_folder + "eb" + str(args.multiplier) + "/decompressed_dat/x_train_0_128X128_size1000.dat",
                dtype=np.float32,
            )
            # x_train_0_noisy = x_train_0_noisy.reshape(1000, 128)
            x_train_0 = np.concatenate((x_train[0].flatten(), x_train_0_noisy), axis=0)
            x_train_0 = x_train_0.reshape(2000, 128)

            x_train = (x_train_0, x_train[1])

            y_train_noisy = np.fromfile(
                input_folder + "eb" + str(args.multiplier) + "/decompressed_dat/y_train_128X128_size1000.dat",
                dtype=np.float32,
            )
            # y_train_noisy = y_train_noisy.reshape(1000, 16384)
            y_train = np.concatenate((y_train.flatten(), y_train_noisy), axis=0).reshape(2000, 16384)

        # Augment train (half clean + half perturbed) => original size ==== mostly clean with some perturbed can we get away by taking significant amount of data as compressed
        if args.augment == "half" and (args.set == "train" and args.subset_train == "both"):
            name = "case5"
            # Assuming you have two equal-size datasets stored in arrays or lists
            data1 = x_train[0].flatten()
            data2 = np.fromfile(
                input_folder + "eb" + str(args.multiplier) + "/decompressed_dat/x_train_0_128X128_size1000.dat",
                dtype=np.float32,
            )

            indices = np.arange(len(data1))
            np.random.shuffle(indices)
            half_length = len(indices) // 2
            indices1 = indices[:half_length]
            indices2 = indices[half_length:]
            indices1.sort()
            indices2.sort()
            x_train_0 = np.zeros_like(data1)
            x_train_0[indices1] = data1[indices1]
            x_train_0[indices2] = data2[indices2]

            x_train = (x_train_0.reshape(1000, 128), x_train[1])

            data3 = y_train.flatten()
            data4 = np.fromfile(
                input_folder + "eb" + str(args.multiplier) + "/decompressed_dat/y_train_128X128_size1000.dat",
                dtype=np.float32,
            )

            indices = np.arange(len(data3))
            np.random.shuffle(indices)
            half_length = len(indices) // 2
            indices3 = indices[:half_length]
            indices4 = indices[half_length:]
            indices3.sort()
            indices4.sort()
            y_train = np.zeros_like(data3)
            y_train[indices3] = data3[indices3]
            y_train[indices4] = data4[indices4]

            y_train = y_train.reshape(1000, 16384)

            # y_train_noisy = np.fromfile(input_folder+"eb"+str(args.multiplier)+"/decompressed_dat/y_train_128X128_size1000.dat.sz3.out", dtype=np.float32)
            # y_train = np.concatenate((y_train.flatten()[:8192000], y_train_noisy[8192000:]), axis=0).reshape(1000, 16384)

    if args.compressor == "zfp":
        data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

        net = dde.maps.DeepONetCartesianProd([nx, 512, 512], [2, 512, 512, 512], "relu", "Glorot normal")
        for trail_number in range(args.trials):
            model = dde.Model(data, net)
            model.compile(
                "adam",
                lr=1e-3,
                decay=("inverse time", 1, 1e-4),
                metrics=["mean l2 relative error"],
            )
            # IC1
            # losshistory, train_state = model.train(epochs=100000, batch_size=None)
            # IC2
            losshistory, train_state = model.train(
                epochs=args.epochs,
                batch_size=16,
                callbacks=(
                    [
                        dde.callbacks.ModelCheckpoint(
                            filepath="results/original_dat_128X128_size1000/ZFP/eb"
                            + str(args.multiplier)
                            + "/trail_"
                            + str(trail_number)
                            + "/best_cases/",
                            save_better_only=True,
                            verbose=1,
                        ),
                        dde.callbacks.EarlyStopping(min_delta=0, patience=10000, baseline=None, monitor="loss_test"),
                    ]
                ),
            )
            y_pred = model.predict(data.test_x)

            result_folder = (
                "results/original_dat_128X128_size1000/ZFP/eb" + str(args.multiplier) + "/trail_" + str(trail_number)
            )
            # np.savetxt(result_folder+"/y_pred_deeponet_"+name+".dat", y_pred[0].reshape(nt, nx))
            # np.savetxt(result_folder+"/y_true_deeponet_"+name+".dat", data.test_y[0].reshape(nt, nx))
            np.savetxt(
                result_folder + "/y_error_deeponet_" + name + ".dat", (y_pred[0] - data.test_y[0]).reshape(nt, nx)
            )

    else:
        data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

        net = dde.maps.DeepONetCartesianProd([nx, 512, 512], [2, 512, 512, 512], "relu", "Glorot normal")
        for trail_number in range(args.trials):
            model = dde.Model(data, net)
            model.compile(
                "adam",
                lr=1e-3,
                decay=("inverse time", 1, 1e-4),
                metrics=["mean l2 relative error"],
            )
            # IC1
            # losshistory, train_state = model.train(epochs=100000, batch_size=None)
            # IC2
            losshistory, train_state = model.train(
                epochs=args.epochs,
                batch_size=16,
                callbacks=(
                    [
                        dde.callbacks.ModelCheckpoint(
                            filepath="results/original_dat_128X128_size1000/ZFP/raw/trail_"
                            + str(trail_number)
                            + "/best_cases/",
                            save_better_only=True,
                            verbose=1,
                        ),
                        dde.callbacks.EarlyStopping(min_delta=0, patience=10000, baseline=None, monitor="loss_test"),
                    ]
                ),
            )
            y_pred = model.predict(data.test_x)
            result_folder = "results/original_dat_128X128_size1000/ZFP/raw/trail_" + str(trail_number)
            # np.savetxt(result_folder+"/y_pred_deeponet_clean.dat", y_pred[0].reshape(nt, nx))
            # np.savetxt(result_folder+"/y_true_deeponet_clean.dat", data.test_y[0].reshape(nt, nx))
            np.savetxt(result_folder + "/y_error_deeponet_clean.dat", (y_pred[0] - data.test_y[0]).reshape(nt, nx))


if __name__ == "__main__":
    main()
