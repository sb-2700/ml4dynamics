import copy
from datetime import datetime

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from box import Box
from jax import random

from ml4dynamics import dynamics

jax.config.update("jax_enable_x64", True)


def main():
  with open(f"config/ns_hit.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  config = Box(config_dict)
  Re = config.sim.Re
  nx = config.sim.nx
  model = dynamics.NS(L=2 * jnp.pi, N=nx, T=10, dt=0.01, nu=1/Re)
  case_num = config.sim.case_num
  step_num = 2000
  patience = 50  # we admit 50 times blow up generations
  writeInterval = 2
  print('Generating NS HIT data with n = {}, Re = {} ...'.format(nx, Re))

  j = 0
  i = 0
  while j < case_num and i < patience:
    i = i + 1
    print('generating the {}-th trajectory...'.format(j))
    model.w_hat = jnp.zeros((model.N, model.N//2+1))
    model.w_hat[:8, :8] = random.normal(random.PRNGKey(0), (8, 8)) * model.init_scale
    model.set_x_hist(model.w_hat, model.CN)

    if not jnp.isnan(model.w_hat).any():
      # successful generating traj
      
      j = j + 1

  if j == case_num:
    U = np.zeros([case_num, step_num // writeInterval, nx, nx, 1])
    U[..., 0] = model.w_hat
    data = {
      "metadata": {
        "type": "ns",
        "t0": 0.0,
        "t1": t,
        "dt": dt * writeInterval,
        "nx": nx,
        "description": "Navier-Stokes PDE dataset",
        "author": "Jiaxi Zhao",
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      },
      "data": {
        "input_fine":
        w.reshape(-1, nx, ny, 2),  # shape [case_num * step_num // writeInterval, nx+2, ny+2, 2]
      },
      "config":
      config_dict,
      "readme":
      "This dataset contains the results of a Reaction-Diffusion PDE solver. "
      "The 'input' field represents the velocity, and the 'output' "
      "field represents the nonlinear RHS."
    }

    with h5py.File(
      f"data/ns/Re{Re}_n{case_num}.h5", "w"
    ) as f:
      metadata_group = f.create_group("metadata")
      for key, value in data["metadata"].items():
        metadata_group.create_dataset(key, data=value)

      data_group = f.create_group("data")
      data_group.create_dataset("inputs", data=data["data"]["input_fine"])
      data_group.create_dataset("outputs", data=data["data"]["output_fine"])
      # data_group.create_dataset(
      #   "input_coarse", data=data["data"]["input_coarse"]
      # )
      # data_group.create_dataset(
      #   "output_coarse", data=data["data"]["output_coarse"]
      # )

      config_group = f.create_group("config")
      config_yaml = yaml.dump(data["config"])
      config_group.attrs["config"] = config_yaml

      f.attrs["readme"] = data["readme"]


if __name__ == "__main__":
  main()
