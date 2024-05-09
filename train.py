import argparse

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

from tensorboardX import SummaryWriter


def train(env_name='humanoid', model_path = 'models/mjx_brax_policy.mjx', writer_path='tensorboard/writer'):
    writer = SummaryWriter(writer_path)

    # instantiate the environment
    env = envs.get_environment(env_name)

    # define the jit reset/step functions
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # initialize the state
    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipeline_state]

    # num_evals is capped at 100
    train_fn = functools.partial(
        ppo.train, num_timesteps=30_000_000, num_evals=100, reward_scaling=0.1,
        episode_length=1000, normalize_observations=True, action_repeat=1,
        unroll_length=10, num_minibatches=32, num_updates_per_batch=8,
        discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=16384,
        batch_size=8192, seed=0)

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]

    max_y, min_y = 13000, 0

    def progress(num_steps, metrics):
        # print(metrics)
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics['eval/episode_reward'])
        ydataerr.append(metrics['eval/episode_reward_std'])

        print(f'{num_steps} steps, reward: {y_data[-1]:.3f} (std: {ydataerr[-1]}), time: {times[-1] - times[-2]}')

        writer.add_scalar('reward', y_data[-1], num_steps)
        # writer.add_scalar('reward_std', ydataerr[-1], num_steps)
        writer.add_scalar('time', (times[-1] - times[-2]).total_seconds(), num_steps)

        # plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
        # plt.ylim([min_y, max_y])

        # plt.xlabel('# environment steps')
        # plt.ylabel('reward per episode')
        # plt.title(f'y={y_data[-1]:.3f}')

        # plt.errorbar(x_data, y_data, yerr=ydataerr)
        # plt.show()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    model.save_params(model_path, params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='humanoid')
    parser.add_argument('--experiment_name', type=str, default='experiment')

    args = parser.parse_args()
    env_name = args.env_name
    experiment_name = args.experiment_name

    train(env_name, f'models/{experiment_name}.mjx', f'tensorboard/{experiment_name}')


if __name__ == '__main__':
    main()
