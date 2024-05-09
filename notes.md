# Docs

https://github.com/google-deepmind/mujoco/discussions/1101

- https://mujoco.readthedocs.io/en/stable/overview.html
- jax memory allocation: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html

# Useful

https://github.com/Denys88/rl_games/blob/master/docs/BRAX.md

# TODO (in priority)

1. see if preallocation=false when gpu is busy is a good idea
2. explore env.render option (locally on mac)
3. interleave rendering and stepping (pass back state from post-rendering into next step)
   - possible to interact with the model during simulation using “perturbations”:
     https://mujoco.readthedocs.io/en/stable/programming/visualization.html#perturbations
   - can probably use mujoco code for visualization, and train policy using mjx
4. store config in a file
