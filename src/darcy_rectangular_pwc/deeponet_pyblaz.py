raise NotImplementedError("I'm still working on this.")
import argparse
import pathlib

import torch
import numpy as np
from scipy import io
from sklearn.preprocessing import StandardScaler
import zfpy
import deepxde as dde
from deepxde.backend import tf

from pyblaz.compression import PyBlaz
from pyblaz.compressed import CompressedTensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_folder", type=str, default="debug")
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--pyblaz_data_folder", type=str, default="../../data/darcy_rectangular_pwc/pyblaz_compressed")
    parser.add_argument("--block_size", type=int, default=4)
    parser.add_argument("--float_type", type=str, default="float32")
    parser.add_argument("--index_dtype", type=str, default="int16")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Append perturbed data to the training set, doubling the original size. Doesn't affect the test set.",
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--epochs", type=int, default=20000, help="The number of batches to train for. Epochs is a bad name."
    )
    parser.add_argument("--patience", type=int, default=5000, help="Early stopping patience.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--decay", type=float, default=1e-4)
    args = parser.parse_args()

    n_train = 1024  # up to 1024
    n_test = 256  # up to 1024
    grid_size = int(((421 - 1) / args.resolution) + 1)
    batch_sizes = args.batch_size  # leave trunk batch size as None.

    data_folder = pathlib.Path("/datasets") / "lulu_cmame2022" / "Darcy_rectangular_PWC"

    results_folder = make_results_folder(args)

    x_train, y_train = get_data(data_folder / "piececonst_r421_N1024_smooth1.mat", n_train, resolution=args.resolution)
    x_test, y_test = get_data(data_folder / "piececonst_r421_N1024_smooth2.mat", n_test, resolution=args.resolution)

    pyblaz_data_folder = (
        pathlib.Path(args.pyblaz_data_folder)
        / f"r{args.resolution}"
        / f"bs{args.block_size}_{args.float_type}_{args.index_dtype}"
    )

    _, trunk_input_train = x_train
    branch_input_train_compressed: CompressedTensor = torch.load(pyblaz_data_folder / "x_train_branch.pyblaz")
    branch_input_test_compressed = torch.load(pyblaz_data_folder / "x_test_branch.pyblaz")

    branch_input_train_coefficientss = branch_input_train_compressed.specified_coefficientss()
    branch_input_test_coefficientss = branch_input_test_compressed.specified_coefficientss()

    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    input_size = 106 * 106 * 16
    activation = "relu"

    for trial_number in range(
        next_trial_number := sum(folder.is_dir() for folder in results_folder.iterdir()),
        next_trial_number + args.trials,
    ):
        trial_folder = results_folder / f"trial_{trial_number}"
        trial_folder.mkdir(parents=True, exist_ok=True)

        branch = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_size,)),
                tf.keras.layers.Reshape((grid_size, grid_size, 1)),
                tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation=activation),
                tf.keras.layers.Conv2D(128, (5, 5), strides=2, activation=activation),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=activation),
                tf.keras.layers.Dense(128),
            ]
        )
        net = dde.maps.DeepONetCartesianProd([input_size, branch], [2, 128, 128, 128, 128], activation, "Glorot normal")

        scaler = StandardScaler().fit(y_train)
        std = np.sqrt(scaler.var_.astype(np.float32))

        def output_transform(inputs, outputs):
            return outputs * std + scaler.mean_.astype(np.float32)

        net.apply_output_transform(output_transform)
        # net.apply_output_transform(dirichlet)

        model = dde.Model(data, net)
        model.compile(
            # tfa.optimizers.AdamW(1e-4, learning_rate=3e-4),
            "adamw",
            lr=args.learning_rate,
            decay=("inverse time", 1, args.decay),
            metrics=["mean l2 relative error"],
        )
        losshistory, train_state = model.train(
            epochs=args.epochs,
            batch_size=batch_sizes,
            callbacks=[
                dde.callbacks.ModelCheckpoint(
                    filepath=trial_folder / "best_parameters",
                    monitor="test loss",
                    save_better_only=True,
                    verbose=1,
                ),
                dde.callbacks.EarlyStopping(min_delta=0, patience=args.patience, baseline=None, monitor="loss_test"),
            ],
        )
        # Hacky way to deal with TensorFlow saving all best weights so far with different names.
        checkpoints = tuple(trial_folder.glob("best_parameters-*"))
        last_checkpoint = max(checkpoints, key=lambda checkpoint: int(checkpoint.name.split(".")[-3].split("-")[-1]))
        last_checkpoint_prefix = ".".join(last_checkpoint.name.split(".")[:-1])
        model.restore(trial_folder / last_checkpoint_prefix)
        print(f"Loaded best weights at {trial_folder / last_checkpoint_prefix}")
        print(f"Cleaning up {len(checkpoints) - 1} checkpoints.")
        for checkpoint in checkpoints:
            if last_checkpoint_prefix not in checkpoint.name:
                checkpoint.unlink()

        predictions = model.predict(x_test)
        errors = (predictions - y_test).reshape(-1, grid_size, grid_size)
        np.save(trial_folder / "errors.npy", errors)


def get_perturb_indices(args, branch_input, y=None):
    perturb_proportion = 0.5 if args.half else 1
    branch_input_perturb_indices = np.random.permutation(branch_input.shape[0])[
        : int(perturb_proportion * branch_input.shape[0])
    ]
    y_perturb_indices = None
    if y is not None:
        y_perturb_indices = np.random.permutation(y.shape[0])[: int(perturb_proportion * y.shape[0])]

    return branch_input_perturb_indices, y_perturb_indices


def make_results_folder(args):
    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    results_folder /= f"r{args.resolution}_pyblaz_" f"bs{args.block_size}_{args.float_type}_{args.index_dtype}"

    results_folder.mkdir(parents=True, exist_ok=True)
    return results_folder


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


def dirichlet(inputs, output):
    x_trunk = inputs[1]
    x, y = x_trunk[:, 0], x_trunk[:, 1]
    return 20 * x * (1 - x) * y * (1 - y) * (output + 1)


def perturb_using_zfp(x_train, y_train, x_test, resolution, std_multiplier):
    """
    Return (x train branch perturbed, y train perturbed, x test branch perturbed).
    Doesn't perturb trunk inputs or test labels.
    """
    # ZFP will make better sense of it if unflattened.
    size = int(((421 - 1) / resolution) + 1)
    x_train_branch = x_train[0].reshape(-1, size, size)
    x_test_branch = x_test[0].reshape(-1, size, size)
    y_train = y_train.reshape(-1, size, size)
    # Shouldn't perturb test labels.

    x_train_branch_perturbed = zfpy.decompress_numpy(
        zfpy.compress_numpy(x_train_branch, tolerance=x_train_branch.std() * std_multiplier)
    ).reshape(-1, size * size)

    y_train_perturbed = zfpy.decompress_numpy(
        zfpy.compress_numpy(y_train, tolerance=y_train.std() * std_multiplier)
    ).reshape(-1, size * size)

    x_test_branch_perturbed = zfpy.decompress_numpy(
        zfpy.compress_numpy(x_test_branch, tolerance=x_test_branch.std() * std_multiplier)
    ).reshape(-1, size * size)

    return (
        x_train_branch_perturbed,
        y_train_perturbed,
        x_test_branch_perturbed,
    )


if __name__ == "__main__":
    main()
