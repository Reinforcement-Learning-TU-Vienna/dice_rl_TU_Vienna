# ---------------------------------------------------------------- #

import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from tf_agents.trajectories.time_step import time_step_spec as get_time_step_spec

from tqdm import tqdm
from abc import ABC, abstractmethod

from gymnasium.wrappers.time_limit import TimeLimit


from dice_rl.data.dataset import EnvStep
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset

from dice_rl_TU_Vienna.specs import get_step_num_spec
from plugins.dice_rl.create_dataset import add_episodes_to_dataset
from plugins.stable_baslines.specs import get_observation_action_spec_from_env

from utils.general import SuppressPrint

# ---------------------------------------------------------------- #

def get_dataset_spec(observation_spec, action_spec, step_num_max, info_specs=None):

    assert observation_spec is not None
    assert action_spec      is not None
    assert step_num_max     is not None

    if info_specs is None: info_specs = {}, {}, {}

    time_step_spec = get_time_step_spec(observation_spec) # type: ignore
    step_num_spec = get_step_num_spec(step_num_max)

    policy_info, env_info, other_info = info_specs

    return EnvStep(
        step_type  =time_step_spec.step_type,
        step_num   =step_num_spec,
        observation=time_step_spec.observation,
        action     =action_spec,
        reward     =time_step_spec.reward,
        discount   =time_step_spec.discount,
        policy_info=policy_info,
        env_info   =env_info,
        other_info =other_info,
    )

# ---------------------------------------------------------------- #

class TFOffpolicyDatasetGenerator(ABC):
    def __init__(
            self,
            num_trajectory, max_trajectory_length,
            by, seed=None):

        self.num_trajectory = num_trajectory
        self.max_trajectory_length = max_trajectory_length

        self.by = by

        self.trajectories = []

        if seed is not None: np.random.seed(seed)

    def get_dataset(self, verbosity=1):
        self.dataset_spec = self.get_dataset_spec()
        self.get_trajectories(verbosity)

        self.dataset_capacity = self.get_capacity()
        self.dataset = TFOffpolicyDataset(
            self.dataset_spec, self.dataset_capacity, ) # type: ignore

        self.add_SEEs_to_dataset(verbosity)

    def get_dataset_spec(self):
        observation_spec, action_spec = self.get_observation_action_spec() # type: ignore
        step_num_max = self.step_num_max
        info_specs = self.get_info_specs()

        return get_dataset_spec(
            observation_spec, action_spec, step_num_max, info_specs)

    @abstractmethod
    def get_capacity(self):
        pass

    @abstractmethod
    def get_observation_action_spec(self):
        pass

    @property
    def step_num_max(self):
        return self.max_trajectory_length

    def get_info_specs(self):
        return {}, {}, {}

    def add_SEEs_to_dataset(self, verbosity=1):
        pbar = range(self.num_trajectory)
        if verbosity > 0:
            print(f"adding {self.by} to dataset")
            pbar = tqdm(pbar)

        for i in pbar:
            trajectory, valid_ids = self.trajectories[i]
            add_episodes_to_dataset(
                episodes=trajectory,
                valid_ids=valid_ids,
                write_dataset=self.dataset,
            )

    def get_trajectories(self, verbosity=1):
        pbar = range(self.num_trajectory)
        if verbosity > 0:
            print(f"getting {self.by}")
            pbar = tqdm(pbar)

        for _ in pbar:
            trajectory = self.get_trajectory()
            self.trajectories.append(trajectory)

    def get_trajectory(self):
        observation, action, reward = \
            self.get_observation_action_reward() # type: ignore

        step_type = self.get_step_type()
        step_num  = self.get_step_num()
        discount  = self.get_discount()

        policy_info, env_info, other_info = self.get_info()

        trajectory = EnvStep(
            step_type, step_num,
            observation, action, reward,
            discount,
            policy_info, env_info, other_info)

        valid_ids = self.get_valid_ids()

        self.last_trajectory_length = None

        return trajectory, valid_ids

    @abstractmethod
    def get_observation_action_reward(self):
        pass

    @abstractmethod
    def get_step_type(self):
        pass

    @abstractmethod
    def get_step_num(self):
        pass

    @abstractmethod
    def get_discount(self):
        pass

    def get_info(self):
        return {}, {}, {}

    @abstractmethod
    def get_valid_ids(self):
        pass


class TFOffpolicyDatasetGenerator_Env(TFOffpolicyDatasetGenerator):
    def __init__(
            self,
            env, get_act,
            num_trajectory, max_trajectory_length,
            by, seed=0):

        self.env = env
        self.get_act = get_act

        super().__init__(num_trajectory, max_trajectory_length, by, seed)

    def get_observation_action_spec(self):
        return get_observation_action_spec_from_env(self.env)


class TFOffpolicyDatasetGenerator_StepsEpisodes(TFOffpolicyDatasetGenerator_Env):
    def __init__(
            self,
            env, get_act, num_trajectory, max_trajectory_length=None,
            by="steps", seed=0, n_pads=0):

        super().__init__(
            env, get_act,
            num_trajectory, max_trajectory_length,
            by, seed)

        if max_trajectory_length is not None:
            env = TimeLimit(env, max_episode_steps=max_trajectory_length)

        self.n_pads = n_pads
        self.i_pad = 0

        self.last_trajectory_length = None

    def get_capacity(self):
        return np.sum([
            np.max(episode.step_num) + 1
                for episode, _ in self.trajectories
        ])

    def get_observation_action_reward(self):
        obs_array = []; act_array = []; rew_array = []

        obs, _ = self.env.reset()

        t = 0
        terminated = False
        truncated = False

        while not self.stop(terminated, truncated):
            act = self.get_act(obs)
            obs_next, rew, terminated, truncated, info = self.env.step(act)

            obs_array.append(obs)
            act_array.append(act)
            rew_array.append(rew)
            t += 1

            obs = obs_next

        self.last_trajectory_length = t

        t_max = self.max_trajectory_length
        if t_max is None or t_max < t:
            self.max_trajectory_length = t_max

        obs_np = np.array(obs_array)
        act_np = np.array(act_array)
        rew_np = np.array(rew_array)

        obs_tf = tf.convert_to_tensor(
            obs_np, dtype=self.dataset_spec.observation.dtype)
        act_tf = tf.convert_to_tensor(
            act_np, dtype=self.dataset_spec.action.dtype)
        rew_tf = tf.convert_to_tensor(
            rew_np, dtype=self.dataset_spec.reward.dtype)

        return obs_tf, act_tf, rew_tf

    def get_step_type(self):
        step_type_np = np.ones(self.last_trajectory_length)
        step_type_np[0] = 0
        step_type_np[-1] = 2
        step_type = tf.convert_to_tensor(
            step_type_np, dtype=self.dataset_spec.step_type.dtype)

        return step_type

    def get_step_num(self):
        step_num_np = np.arange(self.last_trajectory_length)
        step_num = tf.convert_to_tensor(
            step_num_np, dtype=self.dataset_spec.step_num.dtype)

        return step_num

    def get_discount(self):
        discount_np = np.ones(self.last_trajectory_length)
        discount_np[-1] = 0
        discount = tf.convert_to_tensor(
            discount_np, dtype=self.dataset_spec.discount.dtype)

        return discount

    def get_valid_ids(self):
        valid_ids_int = np.ones(self.last_trajectory_length)
        valid_ids = tf.cast(valid_ids_int, dtype=tf.bool)

        return valid_ids

    @property
    def last_trajectory_length(self):
        assert self._last_trajectory_length is not None
        return self._last_trajectory_length

    @last_trajectory_length.setter
    def last_trajectory_length(self, last_trajectory_length):
        self._last_trajectory_length = last_trajectory_length

    def stop(self, terminated, truncated):
        if self.by == "steps":
            done = terminated or truncated
        elif self.by == "episodes":
            done = truncated
        else:
            raise NotImplementedError

        if done:
            if self.i_pad > self.n_pads:
                self.i_pad = 0
                return True
            else:
                self.i_pad += 1
                return False
        else:
            return False


class TFOffpolicyDatasetGenerator_Experience(TFOffpolicyDatasetGenerator_Env):
    def __init__(self, env, num_experience, seed=0):
        get_act = lambda obs: env.action_space.sample()

        num_trajectory = num_experience
        max_trajectory_length = 3

        by = "experience"

        super().__init__(
            env, get_act,
            num_trajectory, max_trajectory_length,
            by, seed=seed,
        )

    def get_capacity(self):
        return self.num_trajectory * self.max_trajectory_length

    def get_observation_action_reward(self):
        obs_init, _ = self.env.reset()
        act_init = self.get_act(obs_init)
        _, rew_init, _, _, _ = self.env.step(act_init)

        obs = self.env.observation_space.sample()
        act = self.get_act(obs)
        self.env.unwrapped.s = obs
        obs_next, rew, _, _, _ = self.env.step(act)
        act_next = self.get_act(obs_next)
        _, rew_next, _, _, _ = self.env.step(act_next)

        obs_list = [obs_init, obs, obs_next]
        act_list = [act_init, act, act_next]
        rew_list = [rew_init, rew, rew_next]

        dtype_obs = self.dataset_spec.observation.dtype
        dtype_act = self.dataset_spec.action.dtype
        dtype_rew = self.dataset_spec.reward.dtype

        obs_tf = tf.convert_to_tensor(obs_list, dtype=dtype_obs)
        act_tf = tf.convert_to_tensor(act_list, dtype=dtype_act)
        rew_tf = tf.convert_to_tensor(rew_list, dtype=dtype_rew)

        return obs_tf, act_tf, rew_tf

    @property
    def step_num_max(self):
        return 2

    def get_step_type(self):
        step_type_np = np.array([0, 1, 1])
        step_type = tf.convert_to_tensor(
            step_type_np, dtype=self.dataset_spec.step_type.dtype)

        return step_type

    def get_step_num(self):
        step_num_np = np.array([0, 1, 2])
        step_num = tf.convert_to_tensor(
            step_num_np, dtype=self.dataset_spec.step_num.dtype)

        return step_num

    def get_discount(self):
        discount_np = np.ones(3)
        discount = tf.convert_to_tensor(
            discount_np, dtype=self.dataset_spec.discount.dtype)

        return discount

    def get_valid_ids(self):
        valid_ids_int = tf.ones(3)
        valid_ids = tf.cast(valid_ids_int, dtype=tf.bool)

        return valid_ids


class TFOffpolicyDatasetGenerator_Dataframe(TFOffpolicyDatasetGenerator):
    def __init__(
            self,
            df, get_split,
            observation_action_spec, n_pads=1):

        self.df = df
        self.get_split = get_split
        self.observation_action_spec = observation_action_spec
        self.n_pads = n_pads

        ids, *_ = self.get_split(self.df)
        u, c = np.unique(ids, return_counts=True)

        self.ids_u = u
        self.ids_c = c
        self.ids_counter = None

        num_trajectory = len(u)
        max_trajectory_length = np.max(c)
        by = "steps"
        seed = None

        super().__init__(num_trajectory, max_trajectory_length, by, seed)

    def get_capacity(self):
        return len(self.df) + self.n_pads * self.num_trajectory

    def get_observation_action_spec(self):
        return self.observation_action_spec

    def get_trajectories(self, verbosity=1):
        self.ids_counter = 0
        super().get_trajectories(verbosity)
        self.ids_counter = None

    def get_trajectory(self):
        trajectory, valid_ids = super().get_trajectory()
        self.increase_ids_counter()
        return trajectory, valid_ids

    @property
    def current_id(self):
        return self.ids_u[self.ids_counter]

    @property
    def current_episode_length(self):
        return np.sum(self.df["id"] == self.current_id)

    @property
    def current_episode_length_padded(self):
        return self.current_episode_length + self.n_pads

    @property
    def current_df_filtered(self):
        return self.df[self.df["id"] == self.current_id]

    def increase_ids_counter(self):
        assert self.ids_counter is not None
        self.ids_counter += 1

    def get_observation_action_reward(self):
        _, _, obs, act, rew = self.get_split(self.current_df_filtered)

        obs = np.array(obs)
        act = np.array(act)
        rew = np.array(rew)

        obs_term = np.expand_dims(obs[-1], axis=0)
        act_term = np.expand_dims(0, axis=0)
        rew_term = np.expand_dims(0, axis=0)

        obs = np.concatenate([obs] + [obs_term] * self.n_pads, axis=0)
        act = np.concatenate([act] + [act_term] * self.n_pads, axis=0)
        rew = np.concatenate([rew] + [rew_term] * self.n_pads, axis=0)

        obs = tf.convert_to_tensor(obs, dtype=self.dataset_spec.observation.dtype)
        act = tf.convert_to_tensor(act, dtype=self.dataset_spec.action     .dtype)
        rew = tf.convert_to_tensor(rew, dtype=self.dataset_spec.reward     .dtype)

        return obs, act, rew

    def get_step_type(self):
        step_type_np = np.ones(self.current_episode_length_padded)
        step_type_np[0] = 0
        step_type_np[-1] = 2
        step_type = tf.convert_to_tensor(
            step_type_np, dtype=self.dataset_spec.step_type.dtype)

        return step_type

    def get_step_num(self):
        step_num_np = np.arange(self.current_episode_length_padded)
        step_num = tf.convert_to_tensor(
            step_num_np, dtype=self.dataset_spec.step_num.dtype)

        return step_num

    def get_discount(self):
        gamma = 1
        discount_np = gamma ** np.arange(self.current_episode_length_padded)
        discount_np[-1] = 0
        discount = tf.convert_to_tensor(
            discount_np, dtype=self.dataset_spec.discount.dtype)

        return discount

    def get_valid_ids(self):
        valid_ids_int = tf.ones(self.current_episode_length_padded)
        valid_ids = tf.cast(valid_ids_int, dtype=tf.bool)

        return valid_ids

# ---------------------------------------------------------------- #

def load_or_create_dataset(dataset_dir, get_generator, verbosity=0):

    try:
        if verbosity == 1:
            print("Try loading dataset", end=" ")
            dataset = TFOffpolicyDataset.load(dataset_dir)
        else:
            with SuppressPrint():
                dataset = TFOffpolicyDataset.load(dataset_dir)
        dataset = TFOffpolicyDataset.load(dataset_dir)

    except KeyboardInterrupt:
        if verbosity == 1: print("KeyboardInterrupt")
        assert False

    except:
        if verbosity == 1: print(); print(f"No dataset found in {dataset_dir}")

        generator = get_generator()
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            generator.get_dataset(verbosity=1)

        if not tf.io.gfile.isdir(dataset_dir):
            tf.io.gfile.makedirs(dataset_dir)

        generator.dataset.save(dataset_dir)

        dataset = generator.dataset

    return dataset


def load_or_create_dataset_StepsEpisodes(
        dataset_dir,
        env, get_act, num_trajectory, max_trajectory_length=None,
        by="steps", seed=0, n_pads=0,
        verbosity=0):
    
    get_generator = lambda: TFOffpolicyDatasetGenerator_StepsEpisodes(
        env, get_act,
        num_trajectory, max_trajectory_length,
        by, seed, n_pads,
    )

    return load_or_create_dataset(dataset_dir, get_generator, verbosity=verbosity)


def load_or_create_dataset_Experience(
        dataset_dir,
        env, num_experience, seed=0,
        verbosity=0):

    get_generator = lambda: TFOffpolicyDatasetGenerator_Experience(
        env, num_experience, seed, )

    return load_or_create_dataset(dataset_dir, get_generator, verbosity=verbosity)


def load_or_create_dataset_Dataframe(
        dataset_dir,
        df, get_split, observation_action_spec, n_pads,
        verbosity=0):

    get_generator = lambda: TFOffpolicyDatasetGenerator_Dataframe(
        df, get_split, observation_action_spec, n_pads, )

    return load_or_create_dataset(dataset_dir, get_generator, verbosity=verbosity)

# ---------------------------------------------------------------- #

def get_all_episodes(dataset, by="steps", verbosity=0):
    if by == "episodes":
        episodes, _ = dataset.get_all_episodes()
        _, num_episodes = episodes.step_type.shape

        episodes = [
            tf.nest.map_structure(lambda t: t[i, ...], episodes)
                for i in range(num_episodes) ]
    
    elif by == "steps":
        all_steps = dataset.get_all_steps(include_terminal_steps=True)

        indices_dataset_init = np.where(all_steps.step_type == 0)[0]
        indices_dataset_term = np.where(all_steps.step_type == 2)[0]

        assert len(indices_dataset_init) == len(indices_dataset_term)

        episodes = []

        A = indices_dataset_init
        B = indices_dataset_term
        total = len(A)
        
        pbar = zip(A, B)
        if verbosity == 1: tqdm(pbar, total=total)
        for k, l in pbar:
            episode = tf.nest.map_structure(lambda t: t[k:l, ...], all_steps)
            episodes.append(episode)

    else:
        raise NotImplementedError

    return episodes

# ---------------------------------------------------------------- #

def goes_to(dataset, obs_from, act_from, return_type="dict"):
    all_steps = dataset.get_all_steps()

    gt = []

    for i in tqdm( range(len(all_steps.step_type)) ):

        obs = all_steps.observation[i]
        act = all_steps.action[i]

        A = obs == obs_from
        B = act == act_from
        if not (A and B): continue

        obs_next = all_steps.observation[i+1]

        gt.append( int(obs_next) )

    gt = sorted(gt)

    if return_type == "list": return gt

    u, c = np.unique(gt, return_counts=True)
    gt = { k: v for k, v in zip(u, c) }

    if return_type == "dict": return gt

# ---------------------------------------------------------------- #

def print_env_step(env_step, blacklist=None, as_df=True):
    if blacklist is None: blacklist = ["policy_info", "env_info", "other_info"]

    if as_df:

        d = {}
        for k, v in env_step._asdict().items():
            if k not in blacklist:
                if tf.rank(v) == 0:
                    d[k] = [ v.numpy() ]
                elif tf.rank(v) == 1:
                    d[k] = v.numpy()
                else:
                    d[k] = [ e.numpy() for e in v ]

        df = pd.DataFrame(d)
        display(df) # type: ignore

    else:

        for k, v in env_step._asdict().items():
            if k not in blacklist:
                print(); print(k); print(v)


def display_dataset_slice(n, dataset, by="steps", m=None):
    if by == "steps":
        all_steps = dataset.get_all_steps(include_terminal_steps=True)
        dataset_slice = tf.nest.map_structure(lambda t: t[:n, ...], all_steps)

    elif by == "episodes":
        all_episodes, _ = dataset.get_all_episodes()
        dataset_slice = tf.nest.map_structure(lambda t: t[:m, :n, ...], all_episodes)

    else: raise NotImplementedError

    print_env_step(dataset_slice)

# ---------------------------------------------------------------- #

def one_hot_encode_observation(env_step, dataset_spec_from, dataset_spec_to):
    args = []

    for k, v in env_step._asdict().items():
        if k == "observation":
            I = np.identity(dataset_spec_from.observation.maximum + 1)
            e = I[v]
            dtype = dataset_spec_to.observation.dtype
            arg = tf.convert_to_tensor(e, dtype=dtype)
        else:
            arg = v

        args.append(arg)

    env_step_OHC_obs = EnvStep(*args)

    return env_step_OHC_obs

# ---------------------------------------------------------------- #
