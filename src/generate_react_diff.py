import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).parent.parent)
sys.path.append(ROOT_PATH)

import argparse
import copy
import pdb

import h5py
import jax
import ml_collections
import numpy as np
import yaml

import utils
from box import Box
from datetime import datetime

np.set_printoptions(precision=15)
jax.config.update("jax_enable_x64", True)


def generate_RD_data(config_dict: ml_collections.ConfigDict):
  """
  Generate reaction-diffusion simulation trajectories.
  Simulating the reaction-diffusion equation over a fine and corase grid with
  pairing initial condition.
  #TODO: we can specify the coarsening ratio in the config file, currently
  # it is fixed to be 2
  """

  config = Box(config_dict)
  print("Generating RD data with gamma = {:.2f}...".format(config.sim.gamma))
  # set simulation parameters
  # warm start, we perform several steps so that the flow comes to a physical
  # state
  warm_up = config.sim.warm_up
  widthx = (config.react_diff.x_right - config.react_diff.x_left)
  widthy = (config.react_diff.y_top - config.react_diff.y_bottom)
  nx = config.react_diff.nx
  ny = config.react_diff.ny
  dx = widthx / nx
  dy = widthy / ny
  gamma = config.react_diff.gamma
  t = config.react_diff.t
  alpha = config.react_diff.alpha
  beta = config.react_diff.beta
  step_num = config.react_diff.nt
  dt = t / step_num

  patience = config.sim.patience  # we admit 5 times blow up generations
  writeInterval = config.sim.writeInterval
  seed = config.sim.seed
  rng = np.random.default_rng(seed)
  case_num = config.sim.case_num

  # simulating training trajectories
  u_coarse = np.zeros((case_num, step_num // writeInterval, nx // 2, ny // 2))
  v_coarse = np.zeros((case_num, step_num // writeInterval, nx // 2, ny // 2))
  labelu_coarse = np.zeros(
    (case_num, step_num // writeInterval, nx // 2, ny // 2)
  )
  labelv_coarse = np.zeros(
    (case_num, step_num // writeInterval, nx // 2, ny // 2)
  )
  u_fine = np.zeros((case_num, step_num // writeInterval, nx, ny))
  v_fine = np.zeros((case_num, step_num // writeInterval, nx, ny))
  labelu_fine = np.zeros((case_num, step_num // writeInterval, nx, ny))
  labelv_fine = np.zeros((case_num, step_num // writeInterval, nx, ny))
  j = 0
  i = 0
  while i < case_num and j < patience:
    print("generating the {}-th trajectory for gamma = {:.2f}".format(i, gamma))
    # simulation in 128x128 grid
    dx = widthx / nx
    u_hist = np.zeros(((step_num + warm_up) // writeInterval, nx, ny))
    v_hist = np.zeros(((step_num + warm_up) // writeInterval, nx, ny))
    utils.assembly_RDmatrix(nx, dt, dx, beta, gamma)
    u_init = rng.normal(size=(nx, ny))
    v_init = rng.normal(size=(nx, ny))

    u = copy.deepcopy(u_init)
    v = copy.deepcopy(v_init)
    u_hist, v_hist, flag = utils.RD_adi(
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
    u_fine[i] = copy.deepcopy(u_hist[warm_up // writeInterval:])
    v_fine[i] = copy.deepcopy(v_hist[warm_up // writeInterval:])
    labelu_fine[i] = u_fine[i] - u_fine[i]**3 - v_fine[i] + alpha
    labelv_fine[i] = beta * (u_fine[i] - v_fine[i])

    u_solu = copy.deepcopy(u)
    v_solu = copy.deepcopy(v)

    # simulation in 64x64 grid
    # averaging the 128-grid to obtain 64 grid initial condition
    tmp = u_init
    u = (tmp[::2, ::2] + tmp[1::2, ::2] + tmp[::2, 1::2] + tmp[1::2, 1::2]) / 4
    tmp = v_init
    v = (tmp[::2, ::2] + tmp[1::2, ::2] + tmp[::2, 1::2] + tmp[1::2, 1::2]) / 4
    dx = widthx / (nx // 2)
    utils.assembly_RDmatrix(nx // 2, dt, dx, beta, gamma)
    u_hist = np.zeros(((step_num + warm_up) // writeInterval, nx // 2, ny // 2))
    v_hist = np.zeros(((step_num + warm_up) // writeInterval, nx // 2, ny // 2))
    u_hist, v_hist, flag = utils.RD_adi(
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
      print("generation fail!")
      continue
    u_coarse[i] = copy.deepcopy(u_hist[warm_up // writeInterval:])
    v_coarse[i] = copy.deepcopy(v_hist[warm_up // writeInterval:])
    labelu_coarse[i] = u_coarse[i] - u_coarse[i]**3 - v_coarse[i] + alpha
    labelv_coarse[i] = beta * (u_coarse[i] - v_coarse[i])
    i = i + 1

  if i == case_num:
    """
    number of successful simulations matchs the goal
    """
    u_coarse = np.concatenate(
      [np.expand_dims(u_coarse, axis=2),
       np.expand_dims(v_coarse, axis=2)],
      axis=2
    )
    label_coarse = np.concatenate(
      [
        np.expand_dims(labelu_coarse, axis=2),
        np.expand_dims(labelv_coarse, axis=2)
      ],
      axis=2
    )
    u_fine = np.concatenate(
      [np.expand_dims(u_fine, axis=2),
       np.expand_dims(v_fine, axis=2)], axis=2
    )
    label_fine = np.concatenate(
      [
        np.expand_dims(labelu_fine, axis=2),
        np.expand_dims(labelv_fine, axis=2)
      ],
      axis=2
    )

    data = {
      "metadata": {
        "type": "react_diff",
        "t0": 0.0,
        "t1": t,
        "dt": dt * writeInterval,
        "nx": nx,
        "ny": ny,
        "description": "Reaction-Diffusion PDE dataset",
        "author": "Jiaxi Zhao",
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      },
      "data": {
        "input_fine":
        u_fine,  # shape [case_num, step_num // writeInterval, 2, nx, ny]
        "output_fine":
        label_fine,  # shape [case_num, step_num // writeInterval, 2, nx, ny]
        "input_coarse":
        u_coarse,  # shape [case_num, step_num // writeInterval, 2, nx//2, ny//2]
        "output_coarse":
        label_coarse,  # shape [case_num, step_num // writeInterval, 2, nx//2, ny//2]
      },
      "config":
      config_dict,
      "readme":
      "This dataset contains the results of a Reaction-Diffusion PDE solver. "
      "The 'input' field represents the velocity, and the 'output' "
      "field represents the nonlinear RHS."
    }

    with h5py.File(
      "data/react_diff/alpha{:.2f}_beta{:.2f}_gamma{:.2f}_n{}".
      format(alpha, beta, gamma, case_num) + ".h5", "w"
    ) as f:
      metadata_group = f.create_group("metadata")
      for key, value in data["metadata"].items():
        metadata_group.create_dataset(key, data=value)

      data_group = f.create_group("data")
      data_group.create_dataset("input_fine", data=data["data"]["input_fine"])
      data_group.create_dataset("output_fine", data=data["data"]["output_fine"])
      data_group.create_dataset(
        "input_coarse", data=data["data"]["input_coarse"]
      )
      data_group.create_dataset(
        "output_coarse", data=data["data"]["output_coarse"]
      )

      config_group = f.create_group("config")
      config_yaml = yaml.dump(data["config"])
      config_group.attrs["config"] = config_yaml

      f.attrs["readme"] = data["readme"]


if __name__ == "__main__":

  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  generate_RD_data(config_dict)
