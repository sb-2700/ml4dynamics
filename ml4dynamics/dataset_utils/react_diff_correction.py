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
  sgs_model = config.train.sgs
  # solver parameters
  rng = random.PRNGKey(config.sim.seed)
  case_num = config.sim.case_num
  # react_diff simulator with periodic BC
  rd_fine, rd_coarse = utils.create_fine_coarse_simulator(config)
  res_fn, _ = dataset_utils.res_int_fn(config_dict)

  if sgs_model == "fine_correction":
    N = n
  elif sgs_model == "filter" or sgs_model == "coarse_correction":
    N = n // r
  inputs = np.zeros((case_num, step_num, N, N, 2))
  outputs = np.zeros((case_num, step_num, N, N, 2))
  for i in range(case_num):
    print(i)
    rng, key = random.split(rng)
    max_freq = 10
    u_fft = jnp.zeros((n, n, 2))
    u_fft = u_fft.at[:max_freq, :max_freq].set(
      random.normal(key, shape=(max_freq, max_freq, 2))
    )
    u0 = jnp.real(jnp.fft.fftn(u_fft, axes=(0, 1))) / n * 10
    rd_fine.run_simulation(u0, rd_fine.adi)

    output = np.zeros_like(outputs[0])
    if sgs_model == "fine_correction":
      calc_correction = jax.jit(partial(
        dataset_utils.calc_correction, rd_fine, rd_coarse, n, r
      ))
      for j in range(rd_fine.step_num):
        output[j] = calc_correction(rd_fine.x_hist[j]) / dt
    else:
      input = jax.vmap(res_fn)(rd_fine.x_hist)
      if sgs_model == "filter":
          output_ = jax.vmap(res_fn)(rd_fine.x_hist**3)
          output[..., 0] = (output_[..., 0] - input[..., 0]**3) /\
            (config.sim.L / n * r)**2
      else:
        for j in range(rd_fine.step_num):
          next_step_fine = rd_fine.adi(rd_fine.x_hist[j])
          next_step_coarse = rd_coarse.adi(input[j])
          if sgs_model == "coarse_correction":
            output[j] = (res_fn(next_step_fine) - next_step_coarse) / dt
    inputs[i] = input
    outputs[i] = output

  inputs = inputs.reshape(-1, N, N, 2)
  outputs = outputs.reshape(-1, N, N, 2)
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
      inputs,  # shape [case_num, step_num // writeInterval, n, n, 2]
      "outputs":
      outputs,  # shape [case_num, step_num // writeInterval, n, n, 2]
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
