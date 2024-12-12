# ---------------------------------------------------------------- #

import os

from dice_rl.data.dataset import Dataset

from plugins.dice_rl.create_dataset import create_dataset

# ---------------------------------------------------------------- #

def get_hparam_str_dataset(env_name, tabular_obs, alpha, seed, num_trajectory, max_trajectory_length):
    return "_".join([
        f"{env_name}",
        f"tabular{tabular_obs}", f"alpha{alpha}", f"seed{seed}",
        f"numtraj{num_trajectory}", f"maxtraj{max_trajectory_length}",
    ])

def load_or_create_dataset(
        datasets_dir, policies_dir,
        env_name, seed, num_trajectory, max_trajectory_length, alpha, tabular_obs):

    hparam_str_dataset = get_hparam_str_dataset(
        env_name, tabular_obs, alpha, seed, num_trajectory, max_trajectory_length)
    dataset_dir = os.path.join(datasets_dir, hparam_str_dataset)

    try:
        dataset = Dataset.load(dataset_dir)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        assert False

    except:
        create_dataset(
            env_name=env_name,
            seed=seed,
            num_trajectory=num_trajectory,
            max_trajectory_length=max_trajectory_length,
            alpha=alpha,
            tabular_obs=tabular_obs,
            save_dir=datasets_dir,
            load_dir=policies_dir,
            force=True,
        )

        dataset = Dataset.load(dataset_dir)

    return dataset

# ---------------------------------------------------------------- #
