from datetime import datetime

import h5py
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import yaml
from box import Box
from matplotlib import pyplot as plt

from ml4dynamics.dataset_utils.dataset_utils import res_int_fn
from ml4dynamics.utils import utils


def main():

  with open(f"config/ks.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  config = Box(config_dict)
  c = config.sim.c
  L = config.sim.L
  T = config.sim.T
  dt = config.sim.dt
  BC = config.sim.BC
  if BC == "periodic":
    N1 = config.sim.n
  elif BC == "Dirichlet-Neumann":
    N1 = config.sim.n - 1
  r = config.sim.r
  sgs_model = config.train.sgs
  N2 = N1 // r
  rng = random.PRNGKey(config.sim.seed)
  case_num = config.sim.case_num
  ks_fine, ks_coarse = utils.create_fine_coarse_simulator(config)
  res_fn, int_fn = res_int_fn(config_dict)
    
  if sgs_model == "fine_correction":
    N = N1
  else:
    N = N2
  inputs = np.zeros((case_num, ks_fine.step_num, N))
  outputs = np.zeros((case_num, ks_fine.step_num, N))
  for i in range(case_num):
    print(i)
    rng, key = random.split(rng)
    # NOTE: the initialization here is important, DO NOT use the random
    # i.i.d. Gaussian noise as the initial condition
    if BC == "periodic":
      dx = ks_fine.L / N1
      u0 = ks_fine.attractor + ks_fine.init_scale * random.normal(key) *\
        jnp.sin(10 * jnp.pi * jnp.linspace(0, L - L/N1, N1) / L)
    elif BC == "Dirichlet-Neumann":
      dx = L / (N1 + 1)
      x = jnp.linspace(dx, L - dx, N1)
      # different choices of initial conditions
      # u0 = ks_fine.attractor + init_scale * random.normal(key) *\
      #   jnp.sin(10 * jnp.pi * x / L)
      # u0 = random.uniform(key) * jnp.sin(8 * jnp.pi * x / 128) +\
      #   random.uniform(rng) * jnp.sin(16 * jnp.pi * x / 128)
      r0 = random.uniform(key) * 20 + 44
      u0 = jnp.exp(-(x - r0)**2 / r0**2 * 4)
    ks_fine.run_simulation(u0, ks_fine.CN_FEM)
    input = jax.vmap(res_fn)(ks_fine.x_hist)[..., 0]  # shape = [step_num, N2]
    output = np.zeros_like(outputs[0])
    if sgs_model == "filter":
      output = (jax.vmap(res_fn)(ks_fine.x_hist**2)[..., 0] - input**2) / 2\
        / ks_coarse.dx**2
    else:
      for j in range(ks_fine.step_num):
        next_step_fine = ks_fine.CN_FEM(ks_fine.x_hist[j])
        next_step_coarse = ks_coarse.CN_FEM(input[j])
        if sgs_model == "coarse_correction":
          output[j] = (res_fn(next_step_fine)[:, 0] - next_step_coarse) / dt
        elif sgs_model == "fine_correction":
          output[j] = (next_step_fine - int_fn(next_step_coarse)[:, 0]) / dt

    if sgs_model == "fine_correction":
      inputs[i] = ks_fine.x_hist
    else:
      inputs[i] = input
    outputs[i] = output

  inputs = inputs.reshape(-1, N)
  outputs = outputs.reshape(-1, N)
  if jnp.any(jnp.isnan(inputs)) or jnp.any(jnp.isnan(outputs)) or\
    jnp.any(jnp.isinf(inputs)) or jnp.any(jnp.isinf(outputs)):
    raise Exception("The data contains Inf or NaN")
  
  plot_ = True
  if plot_:
    inputs_hat = jnp.fft.fft(inputs, axis=-1)
    # inputs_hat = jnp.roll(inputs_hat, inputs_hat.shape[1] // 2, axis=-1)
    inputs_hat = jnp.fft.fftshift(inputs_hat, axes=-1)
    im_array = np.concatenate(
      [inputs.T[None], outputs.T[None], jnp.abs(inputs_hat).T[None]], axis=0
    )
    utils.plot_with_horizontal_colorbar(
      im_array[:, None], fig_size=(10, 4), title_array=None,
      file_path="results/fig/ks.png", dpi=100
    )
    print(ks_coarse.dx**2)
    plt.close()
  data = {
    "metadata": {
      "type": "ks",
      "t0": 0.0,
      "t1": T,
      "dt": dt,
      "n": config.sim.n,
      "BC": BC,
      "description": "Kuramoto–Sivashinsky PDE dataset",
      "author": "Jiaxi Zhao",
      "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
    "data": {
      "inputs":
      inputs[..., None],
      "outputs":
      outputs[..., None],
    },
    "config":
    config_dict,
    "readme":
    "This dataset contains the results of a Kuramoto–Sivashinsky PDE solver. "
    "The 'input' field represents the velocity, and the 'output' "
    "field represents the correction from the coarse grid simulation."
  }

  with h5py.File(
    f"data/ks/c{c:.1f}_T{T}_n{case_num}_{sgs_model}.h5", "w"
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
