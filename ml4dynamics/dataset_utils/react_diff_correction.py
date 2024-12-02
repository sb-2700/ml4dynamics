from datetime import datetime
import h5py
import jax.numpy as jnp
import ml_collections
import torch
import yaml
from box import Box
from jax import random as random

from ml4dynamics.dynamics import RD


def generate_react_diff_correction_dataset(
  config_dict: ml_collections.ConfigDict
):
  
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
  rng = random.PRNGKey(config.sim.seed)
  case_num = config.sim.case_num

  # react_diff simulator with periodic BC
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

  inputs = jnp.zeros((case_num, step_num, 2, nx, nx))
  outputs = jnp.zeros((case_num, step_num, 2, nx, nx))
  for i in range(case_num):
    print(i)
    rng, key = random.split(rng)
    max_freq = 10
    u_fft = jnp.zeros((nx, nx, 2))
    u_fft = u_fft.at[:max_freq, :max_freq].set(
      random.normal(key, shape=(max_freq, max_freq, 2))
    )
    u0 = jnp.real(jnp.fft.fftn(u_fft, axes=(0, 1)).reshape(-1)) / nx
    rd_fine.run_simulation(u0, rd_fine.adi)

    input = rd_fine.x_hist.reshape((step_num, 2, nx, nx))
    output = jnp.zeros_like(input)
    for j in range(rd_fine.step_num):
      next_step_fine = rd_fine.adi(rd_fine.x_hist[i]).reshape(2, nx, nx)
      # NOTE: this is only for downsampling by 2, need to write a general
      # version
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

  inputs = inputs.reshape(-1, 2, nx, nx)
  outputs = outputs.reshape(-1, 2, nx, nx) / dt
  if jnp.any(jnp.isnan(inputs)) or jnp.any(jnp.isnan(outputs)) or\
    jnp.any(jnp.isinf(inputs)) or jnp.any(jnp.isinf(outputs)):
    raise Exception("The data contains Inf or NaN")

  breakpoint()
  data = {
    "metadata": {
      "type": "react_diff",
      "t0": 0.0,
      "t1": T,
      "dt": dt,
      "nx": nx,
      "description": "Reaction-Diffusion PDE dataset",
      "author": "Jiaxi Zhao",
      "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
    "data": {
      "inputs":
      inputs,  # shape [case_num, step_num // writeInterval, 2, nx, ny]
      "outputs":
      outputs,  # shape [case_num, step_num // writeInterval, 2, nx, ny]
    },
    "config":
    config_dict,
    "readme":
    "This dataset contains the results of a Reaction-Diffusion PDE solver. "
    "The 'input' field represents the velocity, and the 'output' "
    "field represents the correction from the coarse grid simulation."
  }

  with h5py.File(
    "data/react_diff/alpha{:.2f}_beta{:.2f}_gamma{:.2f}_n{}.h5".
    format(alpha, beta, gamma, case_num), "w"
  ) as f:
    metadata_group = f.create_group("metadata")
    for key, value in data["metadata"].items():
      metadata_group.create_dataset(key, data=value)

    data_group = f.create_group("data")
    data_group.create_dataset("inputs", data=data["data"]["inputs"])
    data_group.create_dataset("outputs", data=data["data"]["outputs"])

    config_group = f.create_group("config")
    config_yaml = yaml.dump(data["config"])
    config_group.attrs["config"] = config_yaml

    f.attrs["readme"] = data["readme"]
    breakpoint()


if __name__ == "__main__":
  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  generate_react_diff_correction_dataset(config_dict)