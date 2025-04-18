from datetime import datetime

import h5py
import jax
import jax.numpy as jnp
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
  n = config.sim.n
  T = config.sim.T
  dt = config.sim.dt
  L = config.sim.L
  model = dynamics.ns_hit(L=L * jnp.pi, N=n, nu=1/Re, T=T, dt=dt)
  case_num = config.sim.case_num
  patience = 50
  writeInterval = 1
  print('Generating NS HIT data with n = {}, Re = {} ...'.format(n, Re))

  j = 0
  i = 0
  while j < case_num and i < patience:
    i = i + 1
    print('generating the {}-th trajectory...'.format(j))
    model.w_hat = jnp.zeros((model.N, model.N//2+1))
    model.w_hat = model.w_hat.at[:8, :8].set(
      random.normal(random.PRNGKey(0), (8, 8)) * model.init_scale
    )
    model.set_x_hist(model.w_hat, model.CN)

    psi_hat = -model.xhat_hist / model.laplacian_[..., None]
    dpsidx = jnp.fft.irfft2(1j * psi_hat * model.kx[..., None], axes=(0, 1))
    dpsidy = jnp.fft.irfft2(1j * psi_hat * model.ky[..., None], axes=(0, 1))
    dwdx = jnp.fft.irfft2(
      1j * model.xhat_hist * model.kx[..., None], axes=(0, 1)
    )
    dwdy = jnp.fft.irfft2(
      1j * model.xhat_hist * model.ky[..., None], axes=(0, 1)
    )
    J = dpsidy * dwdx - dpsidx * dwdy
    breakpoint()


    if not jnp.isnan(model.x_hist).any():
      # successful generating traj
      j = j + 1

  if j == case_num:
    data = {
      "metadata": {
        "type": "ns",
        "t0": 0.0,
        "t1": T,
        "Re": Re,
        "L": L,
        "dt": dt * writeInterval,
        "nx": n,
        "description": "Navier-Stokes PDE dataset",
        "author": "Jiaxi Zhao",
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      },
      "data": {
        "input_fine":
        model.x_hist.transpose(2, 0, 1)[..., None],  
        # shape [case_num * step_num // writeInterval, nx+2, ny+2, 2]
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
