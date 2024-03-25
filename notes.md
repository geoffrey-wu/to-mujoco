use virtualenv located at /home/geoffrey/envs/to-mujoco

# Installing jaxlib with GPU Support

reference: https://github.com/google/jax/discussions/10323

```bash
pip install jax==0.4.23
pip install jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

- needs to be 0.4.23 or earlier for weird reasons

# Docs

- https://mujoco.readthedocs.io/en/stable/overview.html
- jax memory allocation: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
