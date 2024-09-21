###################################################
#                   finished                      #
###################################################
import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).parent.parent)
sys.path.append(ROOT_PATH)

import argparse
import copy
import pdb

import h5py
import jax
import jax.numpy as jnp
import jax.random as random
import ml_collections
import yaml

import utils
from box import Box

jax.config.update('jax_enable_x64', True)


def generate_RD_data(config: ml_collections.ConfigDict):

  print('Generating RD data with gamma = {:.1f}...'.format(config.gamma))
  # set simulation parameters
  # warm start, we perform several steps so that the flow comes to a physical state
  warm_up = config.warm_up
  widthx = config.widthx
  widthy = config.widthy
  gamma = config.gamma
  dt = config.dt
  alpha = config.alpha
  beta = config.beta
  step_num = 2000
  T = step_num * dt
  patience = 5  # we admit 50 times blow up generations
  writeInterval = 2
  tol = 1e-7
  key = jax.random.key(config.seed)

  # simulating training trajectories
  case_num = 1
  traning_u64 = jnp.zeros([case_num, step_num // writeInterval, 64, 64])
  traning_v64 = jnp.zeros([case_num, step_num // writeInterval, 64, 64])
  traning_labelu64 = jnp.zeros([case_num, step_num // writeInterval, 64, 64])
  traning_labelv64 = jnp.zeros([case_num, step_num // writeInterval, 64, 64])
  traning_u128 = jnp.zeros([case_num, step_num // writeInterval, 128, 128])
  traning_v128 = jnp.zeros([case_num, step_num // writeInterval, 128, 128])
  traning_labelu128 = jnp.zeros([case_num, step_num // writeInterval, 128, 128])
  traning_labelv128 = jnp.zeros([case_num, step_num // writeInterval, 128, 128])
  j = 0
  i = 0
  while i < case_num and j < patience:
    print('generating the {}-th trajectory for gamma = {:.2e}'.format(i, gamma))
    # simulation in 128x128 grid
    key, subkey = jax.random.split(key)
    n = 128
    dx = widthx / n
    u_hist = jnp.zeros([(step_num + warm_up) // writeInterval, n, n])
    v_hist = jnp.zeros([(step_num + warm_up) // writeInterval, n, n])
    utils.assembly_RDmatrix(n, dt, dx, beta, gamma)
    u_init = random.normal(key=subkey, shape=(n, n))
    v_init = random.normal(key=subkey, shape=(n, n))

    u = copy.deepcopy(u_init)
    v = copy.deepcopy(v_init)
    u_hist, v_hist = utils.RD_adi(
      u,
      v,
      dt,
      alpha=alpha,
      beta=beta,
      step_num=step_num + warm_up,
      writeInterval=writeInterval,
    )
    if flag == False:
      j = j + 1
      continue
    traning_u128[i] = copy.deepcopy(u_hist[warm_up // writeInterval:])
    traning_v128[i] = copy.deepcopy(v_hist[warm_up // writeInterval:])
    traning_labelu128[
      i] = traning_u128[i] - traning_u128[i]**3 - traning_v128[i] + alpha
    traning_labelv128[i] = beta * (traning_u128[i] - traning_v128[i])

    u_solu = copy.deepcopy(u)
    v_solu = copy.deepcopy(v)

    # simulation in 64x64 grid
    # averaging the 128-grid to obtain 64 grid initial condition
    tmp = u_init
    u = (tmp[::2, ::2] + tmp[1::2, ::2] + tmp[::2, 1::2] + tmp[1::2, 1::2]) / 4
    tmp = v_init
    v = (tmp[::2, ::2] + tmp[1::2, ::2] + tmp[::2, 1::2] + tmp[1::2, 1::2]) / 4
    n = 64
    dx = widthx / n
    utils.assembly_RDmatrix(n, dt, dx, beta, gamma)
    u_hist = jnp.zeros([(step_num + warm_up) // writeInterval, n, n])
    v_hist = jnp.zeros([(step_num + warm_up) // writeInterval, n, n])
    u_hist, v_hist = utils.RD_adi(
      u,
      v,
      dt,
      alpha=alpha,
      beta=beta,
      step_num=step_num + warm_up,
      writeInterval=writeInterval,
    )
    if flag == False:
      j = j + 1
      continue
    traning_u64[i] = copy.deepcopy(u_hist[warm_up // writeInterval:])
    traning_v64[i] = copy.deepcopy(v_hist[warm_up // writeInterval:])
    traning_labelu64[
      i] = traning_u64[i] - traning_u64[i]**3 - traning_v64[i] + alpha
    traning_labelv64[i] = beta * (traning_u64[i] - traning_v64[i])
    i = i + 1

  if i == case_num:
    # save 64 x 64 data
    U = jnp.concatenate(
      [
        jnp.expand_dims(traning_u64, axis=2),
        jnp.expand_dims(traning_v64, axis=2)
      ],
      axis=2
    )
    label = jnp.concatenate(
      [
        jnp.expand_dims(traning_labelu64, axis=2),
        jnp.expand_dims(traning_labelv64, axis=2)
      ],
      axis=2
    )
    label_dim = 2
    data = {
      "name": 'RD',
      "start_time": 0,
      "end_time": T,
      "write_interval": dt * writeInterval,
      "nx": n,
      "ny": n,
      "input": U,
      "output": label,
      "readme": None
    }

    with h5py.File(
      'data/RD/64-{}.npz'.format(int(gamma * 20)) + '.h5', 'w'
    ) as file:
      for key, value in data.items():
        file.create_dataset(key, data=value)

    # save 128 x 128 data
    n = 128
    U = jnp.concatenate(
      [
        jnp.expand_dims(traning_u128, axis=2),
        jnp.expand_dims(traning_v128, axis=2)
      ],
      axis=2
    )
    label = jnp.concatenate(
      [
        jnp.expand_dims(traning_labelu128, axis=2),
        jnp.expand_dims(traning_labelv128, axis=2)
      ],
      axis=2
    )
    data = {
      "name": 'RD',
      "start_time": 0,
      "end_time": T,
      "write_interval": dt * writeInterval,
      "nx": n,
      "ny": n,
      "input": U,
      "output": label,
      "readme": None
    }

    with h5py.File(
      'data/RD/128-{}.npz'.format(int(gamma * 20)) + '.h5', 'w'
    ) as file:
      for key, value in data.items():
        file.create_dataset(key, data=value)


if __name__ == "__main__":

  with open('config/generate_data.yaml', 'r') as file:
    config_dict = yaml.safe_load(file)
    config = Box(config_dict)
  generate_RD_data(config)
