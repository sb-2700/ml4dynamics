from datetime import datetime
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from box import Box
from jax import random as random

from ml4dynamics import utils
from ml4dynamics.dataset_utils import dataset_utils

jax.config.update("jax_enable_x64", True)


def main():
  with open(f"config/react_diff.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  config = Box(config_dict)
  # model parameters
  alpha = config.sim.alpha
  beta = config.sim.beta
  gamma = config.sim.gamma
  T = config.sim.T
  dt = config.sim.dt
  step_num = int(T / dt)
  n = config.sim.n
  r = config.sim.r
  sgs_model = config.sim.sgs
  # solver parameters
  rng = random.PRNGKey(config.sim.seed)
  case_num = config.sim.case_num
  # react_diff simulator with periodic BC
  rd_fine, rd_coarse = utils.create_fine_coarse_simulator(config)

  if sgs_model == "correction":
    inputs = np.zeros((case_num, step_num, n, n, 2))
    outputs = np.zeros((case_num, step_num, n, n, 2))
  elif sgs_model == "filter":
    inputs = np.zeros((case_num, step_num, n//r, n//r, 2))
    outputs = np.zeros((case_num, step_num, n // r, n // r, 2))
  for i in range(case_num):
    print(i)
    rng, key = random.split(rng)
    max_freq = 10
    u_fft = jnp.zeros((n, n, 2))
    u_fft = u_fft.at[:max_freq, :max_freq].set(
      random.normal(key, shape=(max_freq, max_freq, 2))
    )
    u0 = jnp.real(jnp.fft.fftn(u_fft, axes=(0, 1)).reshape(-1)) / n
    rd_fine.run_simulation(u0, rd_fine.adi)

    if sgs_model == "correction":
      input = rd_fine.x_hist.reshape((step_num, 2, n, n))
      output = np.zeros_like(input)
      calc_correction = jax.jit(partial(
        dataset_utils.calc_correction, rd_fine, rd_coarse, n, r
      ))
      for j in range(rd_fine.step_num):
        output[j] = calc_correction(input[j]) / dt
      input = input.transpose(0, 2, 3, 1)
      output = output.transpose(0, 2, 3, 1)
    elif sgs_model == "filter":
      input = rd_fine.x_hist.reshape((step_num, 2, n, n)).transpose(0, 2, 3, 1)
      input_ = np.zeros((step_num, n // r, n // r, 2))
      output = np.zeros_like(input_)
      for k in range(r):
        for j in range(r):
          input_ += input[:, k::r, j::r]
          output[..., 0] += input[:, k::r, j::r, 0]**3
      input_ = input_ / (r**2)
      output = output / (r**2)
      output[..., 0] = output[..., 0] - input_[..., 0]**3
      input = input
    inputs[i] = input
    outputs[i] = output

  if sgs_model == "correction":
    inputs = inputs.reshape(-1, n, n, 2)
    outputs = outputs.reshape(-1, n, n, 2)
  elif sgs_model == "filter":
    inputs = inputs.reshape(-1, n // r, n // r, 2)
    outputs = outputs.reshape(-1, n // r, n // r, 2) /\
      (config.sim.L / n * r)**2
  if np.any(np.isnan(inputs)) or np.any(np.isnan(outputs)) or\
    np.any(np.isinf(inputs)) or np.any(np.isinf(outputs)):
    raise Exception("The data contains Inf or NaN")

  breakpoint()
  data = {
    "metadata": {
      "type": "react_diff",
      "t0": 0.0,
      "t1": T,
      "dt": dt,
      "n": n,
      "description": "Reaction-Diffusion PDE dataset",
      "author": "Jiaxi Zhao",
      "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
    "data": {
      "inputs":
      inputs,  # shape [case_num, step_num // writeInterval, n, ny, 2]
      "outputs":
      outputs,  # shape [case_num, step_num // writeInterval, n, ny, 2]
    },
    "config":
    config_dict,
    "readme":
    "This dataset contains the results of a Reaction-Diffusion PDE solver. "
    "The 'input' field represents the velocity, and the 'output' "
    "field represents the correction from the coarse grid simulation."
  }

  with h5py.File(
    "data/react_diff/alpha{:.2f}_beta{:.2f}_gamma{:.2f}_n{}_{}.h5".
    format(alpha, beta, gamma, case_num, sgs_model), "w"
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
  main()
