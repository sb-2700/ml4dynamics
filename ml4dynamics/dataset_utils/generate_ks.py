from datetime import datetime

import h5py
import jax.numpy as jnp
import jax.random as random
import yaml
from box import Box
from matplotlib import cm
from matplotlib import pyplot as plt

from ml4dynamics import utils


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
  
  res_op = jnp.zeros((N2, N1))
  int_op = jnp.zeros((N1, N2))
  # for i in range(N2):
  #   res_op = res_op.at[i, i * r:i * r + 7].set(1)
  # res_op /= 7
  for i in range(N2):
    res_op = res_op.at[i, i * r + 1:i * r + 6].set(1)
  res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r + 3].set(0)
  res_op /= 4
  int_op = jnp.linalg.pinv(res_op)
  assert jnp.allclose(res_op @ int_op, jnp.eye(N2))
  assert jnp.allclose(res_op.sum(axis=-1), jnp.ones(N2))
    
  inputs = jnp.zeros((case_num, ks_fine.step_num, N2))
  outputs = jnp.zeros((case_num, ks_fine.step_num, N2))
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
    input = ks_fine.x_hist @ res_op.T  # shape = [step_num, N2]
    output = jnp.zeros_like(input)
    for j in range(ks_fine.step_num):
      if sgs_model == "filter":
        output = ks_fine.x_hist**2 @ res_op.T - input**2
      elif sgs_model == "correction":
        next_step_fine = ks_fine.CN_FEM(
          ks_fine.x_hist[j]
        )  # shape = [N1, step_num]
        next_step_coarse = ks_coarse.CN_FEM(
          input[j]
        )  # shape = [step_num, N2]
        output = output.at[j].set(
          (res_op @ next_step_fine - next_step_coarse) / dt
        )
    inputs = inputs.at[i].set(input)
    outputs = outputs.at[i].set(output)

  inputs = inputs.reshape(-1, N2)
  outputs = outputs.reshape(-1, N2)
  if jnp.any(jnp.isnan(inputs)) or jnp.any(jnp.isnan(outputs)) or\
    jnp.any(jnp.isinf(inputs)) or jnp.any(jnp.isinf(outputs)):
    raise Exception("The data contains Inf or NaN")
  
  plot_ = True
  if plot_:
    plt.figure(figsize=(10, 8))
    plt.subplot(311)
    plt.imshow(inputs.T, cmap=cm.twilight)
    plt.colorbar(orientation="horizontal")
    plt.subplot(312)
    plt.imshow(outputs.T, cmap=cm.twilight)
    plt.colorbar(orientation="horizontal")
    plt.subplot(313)
    inputs_hat = jnp.fft.fft(inputs, axis=-1)
    # inputs_hat = jnp.roll(inputs_hat, inputs_hat.shape[1] // 2, axis=-1)
    inputs_hat = jnp.fft.fftshift(inputs_hat, axes=-1)
    plt.imshow(jnp.abs(inputs_hat).T, cmap=cm.twilight)
    plt.colorbar(orientation="horizontal")
    plt.savefig("ks.png")
  breakpoint()
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