# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

import os
import torch
import shlex
import random
import subprocess
import numpy as np
from isaacgym.torch_utils import torch_rand_float, quat_conjugate, quat_mul, get_euler_xyz
from handem.utils.torch_jit_utils import quat_to_angle_axis, my_quat_rotate

def axis_angle_to_quat(axis, angle):
    """ converts n (axis, angle) tuples to n quaternions """
    ax, ay, az = axis[:, 0], axis[:, 1], axis[:, 2]
    quat = torch.zeros(axis.size(0), 4).to(axis.device)
    qx = ax * torch.sin(angle / 2)
    qy = ay * torch.sin(angle / 2)
    qz = az * torch.sin(angle / 2)
    qw = torch.cos(angle / 2)
    quat[:, 0] = qx
    quat[:, 1] = qy
    quat[:, 2] = qz
    quat[:, 3] = qw
    return quat

def get_euler_disps(quat1, quat2):
    """ computes euler displacements between two quaternions"""
    quat_diff = quat_mul(quat1, quat_conjugate(quat2))
    rpy = list(get_euler_xyz(quat_diff))
    return rpy

def sample_random_quat(n_samples, device):
    """ draw n_samples random quaternions.
    returns (n_samples, 4) tensor """
    alpha = torch.normal(
        mean=0.0,
        std=1.0,
        size=(n_samples, 4)
    ).to(device)
    # restrict first element to be positive to remove double cover
    alpha[:, 0] = torch.abs(alpha[:, 0])    
    alpha_norm = torch.linalg.norm(alpha, dim=1).unsqueeze(-1)
    quat_batch = alpha/alpha_norm
    return quat_batch

def sample_rand_axis_angle(n_samples, device, r=[0, np.pi]):
    """ draw n_samples random axis-angle tuples. """
    axis = torch_rand_float(-1.0, 1.0, (n_samples, 3), device=device)
    axis_norm = torch.linalg.norm(axis, dim=1).unsqueeze(-1)
    axis = axis / axis_norm
    # parse range
    min_angle, max_angle = r[0], r[1]
    angle = torch_rand_float(min_angle, max_angle, (n_samples, 1), device=device)
    return axis, angle

def compute_quat_angle(quat1, quat2):
    # compute angle between two quaternions
    # broadcast quat 1 to quat 2 size
    quat1 = torch.broadcast_to(quat1, quat2.size())
    quat_diff = quat_mul(quat1, quat_conjugate(quat2))
    magnitude, axis = quat_to_angle_axis(quat_diff)
    return torch.abs(magnitude).unsqueeze(1)

def compute_2D_vertex_transform(vertices, pose):
    # vertices: (B, N, 2)
    # pose: (B, 7)
    # returns transformed vertices: (B, N, 2)
    B = vertices.size(0)
    N = vertices.size(1)
    assert pose.size(0) == B
    quat = torch.broadcast_to(
            pose[:, 3:].unsqueeze(1), # unsqueeze to include desired intermediate dim
            (B, N, 4)
    ).reshape(B * N, 4)
    pos = torch.broadcast_to(
            pose[:, :3].unsqueeze(1), # unsqueeze to include desired intermediate dim
            (B, N, 3)
    ).reshape(B * N, 3)
    # lift vertices to be in 3D
    vertices = torch.cat((vertices, torch.zeros((B, N, 1), device=vertices.device)), dim=-1)
    vertices = vertices.reshape(B * N, 3)
    # compute transformed vertices
    transformed_vertices = my_quat_rotate(quat, vertices)
    transformed_vertices = transformed_vertices + pos
    # reshape
    transformed_vertices = transformed_vertices.reshape(B, N, 3)
    return transformed_vertices[:, :, :2]


def euler_to_quat(r, p, y):
    q_x = torch.sin(r / 2) * torch.cos(p / 2) * torch.cos(y / 2) - torch.cos(r / 2) * torch.sin(p / 2) * torch.sin(y / 2)
    q_y = torch.cos(r / 2) * torch.sin(p / 2) * torch.cos(y / 2) + torch.sin(r / 2) * torch.cos(p / 2) * torch.sin(y / 2)
    q_z = torch.cos(r / 2) * torch.cos(p / 2) * torch.sin(y / 2) - torch.sin(r / 2) * torch.sin(p / 2) * torch.cos(y / 2)
    q_w = torch.cos(r / 2) * torch.cos(p / 2) * torch.cos(y / 2) + torch.sin(r / 2) * torch.sin(p / 2) * torch.sin(y / 2)
    quat = torch.stack([q_x, q_y, q_z, q_w], dim=1)
    return quat

def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def git_diff_config(name):
    cmd = f'git diff --unified=0 {name}'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def set_np_formatting():
    """ formats numpy print """
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return seed


class AverageScalarMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean
