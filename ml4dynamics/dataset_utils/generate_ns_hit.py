from datetime import datetime
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import yaml
from box import Box
from jax import random
from jax.lax import conv_general_dilated

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

    kernel_x = 3
    kernel_y = 3
    kernel = jnp.ones((1, kernel_x, kernel_y, 1)) / kernel_x / kernel_y
    conv = partial(
      conv_general_dilated,
      rhs=kernel,
      window_strides=(2, 2),
      padding='VALID',
      dimension_numbers=('NXYC', 'OXYI', 'NXYC'),
    )
    padded_J = jnp.pad(
      J.transpose(2, 0, 1)[..., None],
      pad_width=((0, 0), (1, 1), (1, 1), (0, 0)),
      mode='wrap'
    )
    padded_w = jnp.pad(
      model.x_hist.transpose(2, 0, 1)[..., None],
      pad_width=((0, 0), (1, 1), (1, 1), (0, 0)),
      mode='wrap'
    )
    J_coarse = conv(padded_J)
    w_coarse = conv(padded_w)

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
        "inputs":
        w_coarse,  # shape [case_num * step_num // writeInterval, nx, ny, 1]
        "outputs":
        J_coarse,  # shape [case_num * step_num // writeInterval, nx, ny, 1]
      },
      "config":
      config_dict,
      "readme":
      "This dataset contains the results of a Reaction-Diffusion PDE solver. "
      "The 'input' field represents the velocity, and the 'output' "
      "field represents the nonlinear RHS."
    }

    with h5py.File(
      f"data/ns_hit/Re{Re}_nx{n}_n{case_num}.h5", "w"
    ) as f:
      metadata_group = f.create_group("metadata")
      for key, value in data["metadata"].items():
        metadata_group.create_dataset(key, data=value)

      data_group = f.create_group("data")
      data_group.create_dataset("inputs", data=data["data"]["inputs"])
      data_group.create_dataset("outputs", data=data["data"]["outputs"])
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
