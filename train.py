from envs.humanoid import Humanoid
from envs.ihm_base import IHMBase

from datetime import datetime
import functools
from IPython.display import HTML
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from etils import epath
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx


def train(env_name='humanoid'):
    # instantiate the environment
    env = envs.get_environment(env_name)

    # define the jit reset/step functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # initialize the state
    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipeline_state]

    train_fn = functools.partial(
        ppo.train, num_timesteps=30_000_000, num_evals=5, reward_scaling=0.1,
        episode_length=1000, normalize_observations=True, action_repeat=1,
        unroll_length=10, num_minibatches=32, num_updates_per_batch=8,
        discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=2048,
        batch_size=1024, seed=0)

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]

    max_y, min_y = 13000, 0

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics['eval/episode_reward'])
        ydataerr.append(metrics['eval/episode_reward_std'])

        plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
        plt.ylim([min_y, max_y])

        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title(f'y={y_data[-1]:.3f}')

        plt.errorbar(x_data, y_data, yerr=ydataerr)
        plt.show()

    make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)


if __name__ == '__main__':
    train('ihmbase')
