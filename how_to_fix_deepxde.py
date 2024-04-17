# replace deepxde/nn/tensorflow/deeponet.py SingleOutputStrategy.build with the following code.
def build(self, layer_sizes_branch, layer_sizes_trunk):
    if callable(layer_sizes_branch[1]):
        branch = layer_sizes_branch[1]
    else:
        if layer_sizes_branch[-1] != layer_sizes_trunk[-1]:
            raise AssertionError("Output sizes of branch net and trunk net do not match.")
        branch = self.net.build_branch_net(layer_sizes_branch)
    trunk = self.net.build_trunk_net(layer_sizes_trunk)
    return branch, trunk


# replace deepxde/optimizers/tensorflow/optimizers.py get with the following code.
def get(optimizer, learning_rate=None, decay=None):
    """Retrieves a Keras Optimizer instance."""
    if isinstance(optimizer, tf.keras.optimizers.Optimizer):
        return optimizer
    if is_external_optimizer(optimizer):
        if learning_rate is not None or decay is not None:
            print("Warning: learning rate is ignored for {}".format(optimizer))
        return lbfgs_minimize

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    lr_schedule = _get_learningrate(learning_rate, decay)
    if optimizer == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    if optimizer == "nadam":
        return tf.keras.optimizers.Nadam(learning_rate=lr_schedule)
    if optimizer == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    if optimizer == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr_schedule)

    raise NotImplementedError(f"{optimizer} to be implemented for backend tensorflow.")


# replace deepxde/nn/initializers.py initializer_dict_tf with the following code and import random.
def initializer_dict_tf():
    return {
        "Glorot normal": tf.keras.initializers.glorot_normal(seed=random.getrandbits(32)),
        "Glorot uniform": tf.keras.initializers.glorot_uniform(),
        "He normal": tf.keras.initializers.he_normal(),
        "He uniform": tf.keras.initializers.he_uniform(),
        "LeCun normal": tf.keras.initializers.lecun_normal(),
        "LeCun uniform": tf.keras.initializers.lecun_uniform(),
        "Orthogonal": tf.keras.initializers.Orthogonal(),
        "zeros": tf.zeros_initializer(),
        # Initializers of stacked DeepONet
        "stacked He normal": VarianceScalingStacked(scale=2.0),
        "stacked He uniform": VarianceScalingStacked(scale=2.0, distribution="uniform"),
        "stacked LeCun normal": VarianceScalingStacked(),
        "stacked LeCun uniform": VarianceScalingStacked(distribution="uniform"),
    }
