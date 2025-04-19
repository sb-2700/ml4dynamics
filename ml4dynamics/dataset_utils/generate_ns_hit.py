from datetime import datetime
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import yaml
from box import Box
from jax import random
from jax.lax import conv_general_dilated

from ml4dynamics.utils import create_fine_coarse_simulator

jax.config.update("jax_enable_x64", True)

def main():

  def calc_J(what_hist, model):
    psi_hat = -what_hist / model.laplacian_[None]
    dpsidx = jnp.fft.irfft2(1j * psi_hat * model.kx[None], axes=(1, 2))
    dpsidy = jnp.fft.irfft2(1j * psi_hat * model.ky[None], axes=(1, 2))
    dwdx = jnp.fft.irfft2(
      1j * what_hist * model.kx[None], axes=(1, 2)
    )
    dwdy = jnp.fft.irfft2(
      1j * what_hist * model.ky[None], axes=(1, 2)
    )
    return dpsidy * dwdx - dpsidx * dwdy

  with open(f"config/ns_hit.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  config = Box(config_dict)
  Re = config.sim.Re
  n = config.sim.n
  T = config.sim.T
  dt = config.sim.dt
  L = config.sim.L
  model_fine, model_coarse = create_fine_coarse_simulator(config)
  case_num = config.sim.case_num
  patience = 50
  writeInterval = 1
  print('Generating NS HIT data with n = {}, Re = {} ...'.format(n, Re))

  j = 0
  i = 0
  while j < case_num and i < patience:
    i = i + 1
    print('generating the {}-th trajectory...'.format(j))
    model_fine.w_hat = jnp.zeros((model_fine.N, model_fine.N//2+1))
    f0 = int(jnp.sqrt(n/2)) # init frequency
    model_fine.w_hat = model_fine.w_hat.at[:f0, :f0].set(
      random.normal(random.PRNGKey(0), (f0, f0)) * model_fine.init_scale
    )
    model_fine.set_x_hist(model_fine.w_hat, model_fine.CN)

    kernel_x = 2
    kernel_y = 2
    assert kernel_x == config.sim.r and kernel_y == config.sim.r
    kernel = jnp.ones((1, kernel_x, kernel_y, 1)) / kernel_x / kernel_y
    conv = partial(
      conv_general_dilated,
      rhs=kernel,
      window_strides=(2, 2),
      padding='VALID',
      dimension_numbers=('NXYC', 'OXYI', 'NXYC'),
    )
    periodic_pad = partial(
      jnp.pad, pad_width=(
        (0, 0), (kernel_x//2, kernel_x//2 - 1), 
        (kernel_y//2, kernel_y//2 - 1), (0, 0)
      ),
      mode='wrap'
    )

    padded_w = periodic_pad(model_fine.x_hist[..., None])
    w_coarse = conv(padded_w)
    J = calc_J(model_fine.xhat_hist, model_fine)
    J_coarse = calc_J(
      jnp.fft.rfft2(w_coarse[..., 0], axes=(1, 2)), model_coarse
    )[..., None]
    padded_J = periodic_pad(J[..., None])
    J_filter = conv(padded_J)
    tau = (J_filter - J_coarse) / model_coarse.dx**2
    breakpoint()

    if not tau.shape[1] == tau.shape[2] == model_coarse.N:
      breakpoint()
      raise Exception("The shape of tau is wrong.")

    if not jnp.isnan(tau).any() and not jnp.isinf(w_coarse).any():
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
        tau,  # shape [case_num * step_num // writeInterval, nx, ny, 1]
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
