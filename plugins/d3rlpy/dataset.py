# ---------------------------------------------------------------- #

import numpy as np

from d3rlpy.dataset import MDPDataset

# ---------------------------------------------------------------- #

def convert_dataset(dataset):
    observations = np.array( dataset["obs"] ); observations = np.expand_dims(observations, axis=-1)
    actions      = np.array( dataset["act"] )
    rewards      = np.array( dataset["rew"] ); rewards = np.roll(rewards, shift=1)
    terminals    = np.array( dataset.groupby("id")["t"].transform(lambda x: x.index == x.index[-1]) )

    return MDPDataset(observations, actions, rewards, terminals)

# ---------------------------------------------------------------- #
