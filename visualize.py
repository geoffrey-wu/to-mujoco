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


def visualize(env_name='humanoid', model_path='models/mjx_brax_policy.mjx', render_path='html_renders/render.html'):
    env = envs.get_environment(env_name)

    def get_make_inference_fn():
        ppo_network = ppo_networks.make_ppo_networks(
            env.observation_size, env.action_size)
        make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
        return make_inference_fn

    make_inference_fn = get_make_inference_fn()
    params = model.load_params(model_path)

    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # initialize the state
    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipeline_state]

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    # grab a trajectory
    n_steps = 500
    render_every = 2

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

        if state.done:
            break

    # media.show_video(env.render(rollout[::render_every], camera='side'), fps=1.0 / env.dt / render_every)

    with open(render_path, 'w') as f:
        f.write(HTML(html.render(env.sys.replace(dt=env.dt), rollout)).data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='humanoid')
    parser.add_argument('--experiment_name', type=str, default='experiment')

    args = parser.parse_args()
    env_name = args.env_name
    experiment_name = args.experiment_name

    visualize(env_name, f'models/{experiment_name}.mjx', render_path=f'html_renders/{experiment_name}.html')


if __name__ == '__main__':
    main()
