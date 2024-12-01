from functools import partial
from typing import Iterator, Optional, Tuple

import jax.numpy as jnp
import ml_collections
import torch
import yaml
from box import Box
from jax import random as random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ml4dynamics.dynamics import RD
from ml4dynamics.models.models import EDNet, UNet
from ml4dynamics.types import Batch, OptState, PRNGKey


def main(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  # model parameters
  warm_up = config.sim.warm_up
  alpha = config.react_diff.alpha
  beta = config.react_diff.beta
  gamma = config.react_diff.gamma
  d = config.react_diff.d
  T = config.react_diff.T
  dt = config.react_diff.dt
  step_num = int(T / dt)
  Lx = config.react_diff.Lx
  nx = config.react_diff.nx
  dx = Lx / nx
  r = config.react_diff.r
  # solver parameters
  dagger_epochs = config.train.dagger_epochs
  rng = random.PRNGKey(config.sim.seed)
  case_num = config.sim.case_num

  rd_fine = RD(
    N=nx**2 * 2,
    T=T,
    dt=dt,
    dx=dx,
    tol=1e-8,
    init_scale=4,
    tv_scale=1e-8,
    L=Lx,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    d=d,
    device=torch.device('cpu'),
  )

  rd_coarse = RD(
    N=(nx//r)**2 * 2,
    T=T,
    dt=dt,
    dx=dx * r,
    tol=1e-8,
    init_scale=4,
    tv_scale=1e-8,
    L=Lx,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    d=d,
    device=torch.device('cpu'),
  )

  def run_simulation(uv: jnp.ndarray):
    step_num = rd_fine.step_num
    x_hist = jnp.zeros([step_num, 2, nx, nx])
    for i in range(step_num):
      x_hist = x_hist.at[i].set(uv)
      correction = model(uv)
      tmp = (uv[:, 0::2, 0::2] + uv[:, 1::2, 0::2] +
        uv[:, 0::2, 1::2] + uv[:, 1::2, 1::2]) / 4
      uv = rd_coarse.adi(tmp.reshape(-1)).reshape(2, nx // r, nx // r)
      uv = jnp.vstack(
        [jnp.kron(uv[0], jnp.ones((r, r))).reshape(1, nx, nx),
         jnp.kron(uv[1], jnp.ones((r, r))).reshape(1, nx, nx),]
      )
      uv += correction

    return x_hist

  iters = tqdm(range(dagger_epochs))
  for i in iters:
    train()

    # DAgger step
    x_hist = run_simulation(uv)
    input = rd_fine.x_hist.reshape((step_num, 2, nx, nx))
    output = jnp.zeros_like(input)
    for j in range(x_hist.shape[0]):
      next_step_fine = rd_fine.adi(x_hist[i]).reshape(2, nx, nx)
      tmp = (input[i, :, 0::2, 0::2] + input[i, :, 1::2, 0::2] +
        input[i, :, 0::2, 1::2] + input[i, :, 1::2, 1::2]) / 4
      uv = tmp.reshape(-1)
      next_step_coarse = rd_coarse.adi(uv).reshape(2, nx // r, nx // r)
      next_steo_coarse_interp = jnp.vstack(
        [jnp.kron(next_step_coarse[0], jnp.ones((r, r))).reshape(1, nx, nx),
         jnp.kron(next_step_coarse[1], jnp.ones((r, r))).reshape(1, nx, nx),]
      )
      output = output.at[j].set(next_step_fine - next_steo_coarse_interp)
    inputs = inputs.at[i].set(input)
    outputs = outputs.at[i].set(output)



if __name__ == "__main__":

  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
