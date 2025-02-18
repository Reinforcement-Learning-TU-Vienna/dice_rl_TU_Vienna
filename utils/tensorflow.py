# ---------------------------------------------------------------- #

from tensorflow.keras.optimizers.schedules import ( # type: ignore
    ExponentialDecay, PiecewiseConstantDecay, )

# ---------------------------------------------------------------- #

def learning_rate_hyperparameter(lr):
    if isinstance(lr, ExponentialDecay):
        hyperparameter = {
            "name": "ExponentialDecay",
            "value": {
                "initial_learning_rate": lr.initial_learning_rate,
                "decay_steps": lr.decay_steps,
                "decay_rate": lr.decay_rate,
                "staircase": lr.staircase,
            }
        }

    elif isinstance(lr, PiecewiseConstantDecay):
        hyperparameter = {
            "name": "PiecewiseConstantDecay",
            "value": {
                "boundaries": lr.boundaries,
                "values": lr.values,
            }
        }

    else:
        hyperparameter = lr

    return hyperparameter

# ---------------------------------------------------------------- #
