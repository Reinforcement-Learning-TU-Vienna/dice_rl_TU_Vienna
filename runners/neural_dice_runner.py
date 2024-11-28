# ---------------------------------------------------------------- #

import os

import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
from tf_agents.policies import tf_policy
from tensorflow.keras.optimizers import SGD # type: ignore
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay # type: ignore
from tensorflow.keras.utils import set_random_seed # type: ignore

from tqdm import tqdm
from typing import List

from dice_rl.networks.value_network import ValueNetwork
from dice_rl.data.dataset import OffpolicyDataset, EnvStep
from dice_rl.utils.common import reverse_broadcast

from dice_rl_TU_Vienna.sample import get_probs
from dice_rl_TU_Vienna.estimators.estimator import get_fullbatch_average
from dice_rl_TU_Vienna.estimators.neural.neural_dice import NeuralDice

# ---------------------------------------------------------------- #

def lr_to_str(lr):

    if isinstance(lr, ExponentialDecay):
        a = lr.initial_learning_rate
        b = lr.decay_steps
        c = lr.decay_rate
        d = lr.staircase
        lr_str = f"ExpDec({a}, {b}, {c}, {d})"

    elif isinstance(lr, PiecewiseConstantDecay):
        a = lr.boundaries
        b = lr.values
        lr_str = f"PCDec({a}, {b})"

    else:
        lr_str = str(lr)

    return lr_str

# ---------------------------------------------------------------- #

class NeuralDiceRunner(ABC):
    error_names_pbar = ["pv_error_s", "pv_error_w", "sdc_L2_error"]
    save_interval = 100

    @property
    @abstractmethod
    def __name__(self) -> str: pass

    def __init__(
            self,
            gamma, n_step, batch_size, seed,
            hidden_dims_primal, hidden_dims_dual,
            learning_rate_primal, learning_rate_dual,
            regularizer_primal, regularizer_dual,
            dataset, dataset_spec=None, target_policy=None,
            save_dir=None,
            by="steps", analytical_solver=None,
            env_step_preprocessing=None, aux_recorder=None, aux_recorder_pbar=None,
            verbosity=1):

        if seed is not None: set_random_seed(seed)

        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.seed = seed

        self.hidden_dims_primal = hidden_dims_primal
        self.hidden_dims_dual   = hidden_dims_dual

        self.learning_rate_primal = learning_rate_primal
        self.learning_rate_dual   = learning_rate_dual

        self.regularizer_primal = regularizer_primal
        self.regularizer_dual   = regularizer_dual

        self.dataset = dataset
        self.dataset_spec = dataset_spec if dataset_spec is not None else dataset.spec
        self.target_policy = target_policy

        self.by = by
        self.analytical_solver = analytical_solver

        self.env_step_preprocessing = env_step_preprocessing
        self.aux_recorder = aux_recorder
        self.aux_recorder_pbar = aux_recorder_pbar

        self.verbosity = verbosity

        self.set_up_parameter_recording(save_dir)
        self.set_up_estimator()
        self.set_up_exact_solution()

        self.init_mask = self.dataset.get_all_steps(include_terminal_steps=True).step_type == 0
        self.n_init_step = int( tf.reduce_sum( tf.cast(self.init_mask, tf.int64) ) )

        self.eval_loop()

    def set_up_parameter_recording(self, save_dir):
        if save_dir is not None:
            self.save_dir = os.path.join(
                save_dir,
                self.__name__, self.hparam_str_evaluation, )

            summary_writer = tf.summary.create_file_writer(logdir=self.save_dir)
            summary_writer.set_as_default()

        else:
            self.save_dir = None
            tf.summary.create_noop_writer()

        self.set_up_exact_solution()

    def set_up_exact_solution(self):
        sdc_exact, pv_exact = None, None
        if self.analytical_solver is not None:
            pv_exact, sdc_exact, _ = self.analytical_solver.solve(
                self.gamma, primal_dual="dual")

        self.pv_exact  = pv_exact
        self.sdc_exact = sdc_exact

    @property
    def hparam_str_evaluation(self):
        lrp = lr_to_str(self.learning_rate_primal)
        lrd = lr_to_str(self.learning_rate_dual)

        return "_".join([
            f"gam{self.gamma}",
            f"batchs{self.batch_size}", f"seed{self.seed}",
            f"hdp{self.hidden_dims_primal}", f"hdd{self.hidden_dims_dual}",
            f"lrp{lrp}", f"lrd{lrd}",
            f"regp{self.regularizer_primal}", f"regd{self.regularizer_dual}",
        ])

    def set_up_estimator(self):
        input_tensor_spec = self.dataset_spec.observation, self.dataset_spec.action

        network_primal = ValueNetwork(
            input_tensor_spec,
            fc_layer_params=self.hidden_dims_primal,
        )

        network_dual = ValueNetwork(
            input_tensor_spec,
            fc_layer_params=self.hidden_dims_dual,
            output_activation_fn=self.dual_output_activation_fn,
        )

        optimizer_primal = SGD(self.learning_rate_primal)
        optimizer_dual   = SGD(self.learning_rate_dual)

        self.estimator: NeuralDice

        return (
            network_primal, network_dual,
            optimizer_primal, optimizer_dual,
        )

    @property
    def dual_output_activation_fn(self): return tf.identity

    def get_errors(self, pv_approx, sdc_approx_network) -> dict:
        errors = {}

        if self.analytical_solver is not None:
            errors = self.analytical_solver.errors(
                gamma=self.gamma,
                pv_approx=pv_approx, sdc_approx_network=sdc_approx_network,
                pv_exact=self.pv_exact, sdc_exact=self.sdc_exact,
            )

        return {
            k: tf.convert_to_tensor(v,  dtype=tf.float32)
                for k, v in errors.items()
        }

    def eval_loop(self):
        self.i_step_tf = tf.Variable(0, dtype=tf.int64)
        tf.summary.experimental.set_step(self.i_step_tf)

        pbar = range(self.n_step)
        if self.verbosity > 0:
            print(self.__name__)
            print(self.hparam_str_evaluation)
            pbar = tqdm(pbar)

        for i_step in pbar:
            env_steps = self.get_sample()

            values, loss, gradients = self.estimator.eval_step(
                *env_steps, self.batch_size, self.target_policy, )

            mean_loss = np.nanmean(loss)
            if np.isnan(mean_loss): break

            if i_step % self.save_interval == 0 or i_step == self.n_step - 1:
                d = {}

                tf.summary.scalar("loss", mean_loss)
                d["loss"] = float( mean_loss )

                pv_s = self.estimate_average_reward(
                    weighted=False, dataset=self.dataset, target_policy=self.target_policy)
                pv_w = self.estimate_average_reward(
                    weighted=True,  dataset=self.dataset, target_policy=self.target_policy)

                tf.summary.scalar("pv_s", pv_s)
                d["pv_s"] = float(pv_s) # type: ignore
                tf.summary.scalar("pv_w", pv_w)
                d["pv_w"] = float(pv_w) # type: ignore

                if self.aux_recorder is not None:
                    aux_recordings = self.aux_recorder(
                        self.estimator,
                        env_steps, values, loss, gradients, )

                    for k, v in aux_recordings.items():
                        tf.summary.scalar(k, v)

                    if self.aux_recorder_pbar is not None:
                        for k in self.aux_recorder_pbar:
                            d[k] = aux_recordings[k]

                if self.analytical_solver is not None:
                    errors = self.get_errors(
                        pv_approx={ "s": pv_s, "w": pv_w, },
                        sdc_approx_network=self.estimator.network_dual,
                    )

                    for k, v in errors.items():
                        tf.summary.scalar(k, v)

                    for error_name in self.error_names_pbar:
                        d[error_name] = float( errors[error_name] )

                if self.verbosity > 0:
                    pbar.set_postfix(d) # type: ignore

            self.i_step_tf.assign_add(1)

    def get_sample(self) -> Tuple[EnvStep, EnvStep, EnvStep]:

        if self.by == "episodes":
            s, _ = self.dataset.get_episode(self.batch_size, truncate_episode_at=1 )
            env_step_init = tf.nest.map_structure(lambda t: t[:, 0, ...], s)

            s = self.dataset.get_step(self.batch_size, num_steps=2)

            env_step_this = tf.nest.map_structure(lambda t: t[:, 0, ...], s)
            env_step_next = tf.nest.map_structure(lambda t: t[:, 1, ...], s)

        elif self.by == "steps":
            m = self.n_init_step
            indices = tf.random.uniform(shape=(self.batch_size,), minval=0, maxval=m, dtype=tf.int64)

            env_step_init = tf.nest.map_structure(
                lambda t: tf.gather(t[self.init_mask], indices),
                self.dataset.get_all_steps(include_terminal_steps=True) )

            s = self.dataset.get_step(self.batch_size, num_steps=2)

            env_step_this = tf.nest.map_structure(lambda t: t[:, 0, ...], s)
            env_step_next = tf.nest.map_structure(lambda t: t[:, 1, ...], s)

        elif self.by == "experience":
            m = self.dataset.capacity / 3
            indices = tf.random.uniform(shape=(self.batch_size,), minval=0, maxval=m, dtype=tf.int64 )

            s = tf.nest.map_structure(lambda t: tf.gather(t, indices), self.dataset.get_all_episodes()[0])

            env_step_init = tf.nest.map_structure(lambda t: t[:, 0, ...], s)
            env_step_this = tf.nest.map_structure(lambda t: t[:, 1, ...], s)
            env_step_next = tf.nest.map_structure(lambda t: t[:, 2, ...], s)

        else:
            raise NotImplementedError

        env_step_init = self.preprocess_env_step(env_step_init)
        env_step_this = self.preprocess_env_step(env_step_this)
        env_step_next = self.preprocess_env_step(env_step_next)

        return env_step_init, env_step_this, env_step_next

    def preprocess_env_step(self, env_step):
        if self.env_step_preprocessing is None:
            env_step_preprocessed = env_step

        else:
            assert callable(self.env_step_preprocessing)
            env_step_preprocessed = self.env_step_preprocessing(
                env_step, self.dataset.spec, self.dataset_spec)

        assert isinstance(env_step_preprocessed, EnvStep)
        return env_step_preprocessed

    def estimate_average_reward(
        self,
        weighted,
        dataset: OffpolicyDataset,
        target_policy: Union[tf_policy.TFPolicy, None] = None):

        def weight_fn(env_step):
            env_step_preprocessed = self.preprocess_env_step(env_step)

            dual_value = self.estimator.get_value(
                self.estimator.network_dual, env_step_preprocessed, )

            policy_ratio = 1.0
            if not self.estimator.solve_for_state_action_ratio:
                A = get_probs(env_step, target_policy)
                B = env_step.get_log_probability()
                policy_ratio = tf.exp(A - B)

            dual_value *= reverse_broadcast(
                policy_ratio, dual_value)

            return dual_value

        reward_fn = self.estimator.reward_fn

        pv = get_fullbatch_average(
            dataset,
            limit=None,
            by="steps" if self.by == "episodes" else self.by,
            reward_fn=reward_fn,
            weight_fn=weight_fn,
            weighted=weighted,
        )

        return pv

# ---------------------------------------------------------------- #
