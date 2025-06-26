import copy
from datetime import datetime
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as r
import yaml
from box import Box

import ml4dynamics.utils as utils

jax.config.update("jax_enable_x64", True)


def main():
  with open(f"config/ns_channel.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  config = Box(config_dict)
  nx = config.sim.nx
  ny = config.sim.ny
  Re = config.sim.Re
  dx = 1 / ny
  dy = 1 / ny
  BC = config.sim.BC
  case_num = config.sim.case_num
  T = config.sim.T
  dt = config.sim.dt
  step_num = int(T / dt)
  patience = 50  # we admit 50 times blow up generations
  warm_up = 500
  writeInterval = 1
  r.seed(0)
  print('Generating channel NS data with nx = {}, Re = {} ...'.format(nx, Re))

  utils.assembly_NSmatrix(nx, ny, dx, dy, BC)
  u_hist_ = np.zeros([case_num, step_num // writeInterval, nx, ny])
  v_hist_ = np.zeros([case_num, step_num // writeInterval, nx, ny])
  p_hist_ = np.zeros([case_num, step_num // writeInterval, nx, ny])

  def inlet(y: jnp.ndarray):
    """set the inlet velocity"""
    return y * (1 - y) * jnp.exp(-10 * (y - y0)**2) * 3

  j = 0
  i = 0
  while j < case_num and i < patience:
    i = i + 1
    print('generating the {}-th trajectory...'.format(j))
    y0 = 0.325
    iter = partial(
      utils.projection_correction,
      dx=dx,
      dy=dy,
      nx=nx,
      ny=ny,
      dt=dt,
      Re=Re,
      BC=BC
    )
    u_inlet = inlet(np.linspace(dy / 2, 1 - dy / 2, ny))
    u = jnp.tile(u_inlet, (nx, 1))
    v = jnp.zeros([nx, ny])
    p = jnp.zeros(
      [nx, ny]
    )  # staggered grid, the size of grid p is undetermined
    # source term in poisson equation: divergence of the predicted velocity field
    # divu = np.zeros(
    #   [nx, ny]
    # )
    u_hist = np.zeros([(step_num + warm_up) // writeInterval, nx, ny])
    v_hist = np.zeros([(step_num + warm_up) // writeInterval, nx, ny])
    p_hist = np.zeros([(step_num + warm_up) // writeInterval, nx, ny])

    for k in range(step_num + warm_up):
      t = k * dt
      u, v, p = iter(u, v, p, t)
      if k % writeInterval == 0:
        u_hist[k // writeInterval, :, :] = copy.deepcopy(u)
        v_hist[k // writeInterval, :, :] = copy.deepcopy(v)
        p_hist[k // writeInterval, :, :] = copy.deepcopy(p)

    if not np.isnan(u_hist).any() and not np.isnan(v_hist).any(
    ) and not np.isnan(p_hist).any():
      # successful generating traj
      u_hist_[j] = copy.deepcopy(u_hist[warm_up // writeInterval:])
      v_hist_[j] = copy.deepcopy(v_hist[warm_up // writeInterval:])
      p_hist_[j] = copy.deepcopy(p_hist[warm_up // writeInterval:])
      j = j + 1

  if j == case_num:
    U = np.zeros([case_num, step_num // writeInterval, nx, ny, 2])
    U[..., 0] = u_hist_
    U[..., 1] = v_hist_
    data = {
      "metadata": {
        "type": "ns",
        "t0": 0.0,
        "t1": t,
        "dt": dt * writeInterval,
        "nx": nx,
        "ny": ny,
        "description": "Navier-Stokes PDE dataset",
        "author": "Jiaxi Zhao",
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      },
      "data": {
        "inputs": U.reshape(
          -1, nx, ny, 2
        ),  # shape [case_num * step_num // writeInterval, nx+2, ny+2, 2]
        "outputs": p_hist_.reshape(
          -1, nx, ny, 1
        ),  # shape [case_num * step_num // writeInterval, nx, ny, 1]
      },
      "config":
      config_dict,
      "readme":
      "This dataset contains the results of a Reaction-Diffusion PDE solver. "
      "The 'input' field represents the velocity, and the 'output' "
      "field represents the nonlinear RHS."
    }

    with h5py.File(
      f"data/ns_channel/{BC}_Re{Re}_nx{nx}_n{case_num}.h5", "w"
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


if __name__ == "__main__":
  main()
