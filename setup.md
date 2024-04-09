# Linux with CUDA

If possible, use virtualenv located at `/home/geoffrey/envs/to-mujoco`.

1. Create a virtual environment **using python3.9 or newer**:

```bash
python3.9 -m venv ~/envs/to-mujoco
source ~/envs/to-mujoco/bin/activate
```

2. Install jaxlib with GPU Support
    - Reference: https://github.com/google/jax/discussions/10323
    - Needs to be 0.4.23 or earlier for weird reasons

```bash
pip install jax==0.4.23
pip install jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

# Mac with M1

0. Set up mounting: https://apple.stackexchange.com/questions/5209/how-can-i-mount-sftp-ssh-in-finder-on-os-x-snow-leopard

1. Ensure you are using a conda build natively built for M1
    - It must be conda; see https://stackoverflow.com/questions/68327863/importing-jax-fails-on-mac-with-m1-chip
    - It must not be emulation of x86_64: see https://github.com/openai/mujoco-py/issues/662

https://docs.anaconda.com/free/miniconda/

2. Create a python3.9 environment

```bash
conda create --name to-mujoco python=3.9
conda activate to-mujoco
```

3. Install jax and jaxlib

```bash
conda install -c conda-forge jaxlib
conda install -c conda-forge jax
```

4. Install brax

```bash
pip install brax
```

5. Install everything else

```bash
conda install matplotlib mediapy tensorboardX
```
