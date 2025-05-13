from datetime import datetime
from functools import partial

import h5py
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import yaml
from box import Box
from jax.lax import conv_general_dilated

from ml4dynamics.dataset_utils import dataset_utils
from ml4dynamics import utils


def main():

  # @jax.jit
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
  T = config.sim.T
  dt = config.sim.dt
  L = config.sim.L
  r = config.sim.r
  sgs_model = config.train.sgs
  case_num = config.sim.case_num
  patience = 50
  writeInterval = 1
  model_fine, model_coarse = utils.create_fine_coarse_simulator(config)
  # model_fine.nu = 0
  # model_coarse.nu = 0
  n = model_coarse.N
  u_ = np.zeros((case_num, int(T/dt), n, n, 1))
  outputs = np.zeros((case_num, int(T/dt), n, n, 1))
  print('Generating NS HIT data with n = {}, Re = {} ...'.format(n, Re))

  j = 0
  i = 0
  while j < case_num and i < patience:
    i = i + 1
    print('generating the {}-th trajectory...'.format(j))
    model_fine.w_hat = utils.hit_init_cond("spec_random", model_fine)
    model_fine.set_x_hist(model_fine.w_hat, model_fine.CN)
    res_fn, _ = dataset_utils.res_int_fn(config_dict)

    if sgs_model == "filter":
      kernel_x = kernel_y = r
      kernel = jnp.ones((1, kernel_x, kernel_y, 1)) / kernel_x / kernel_y
      conv = partial(
        conv_general_dilated,
        rhs=kernel,
        window_strides=(r, r),
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
      # w_coarse = jax.vmap(res_fn)(model_fine.x_hist[..., None])
      J = calc_J(model_fine.xhat_hist, model_fine)
      J_coarse = calc_J(
        jnp.fft.rfft2(w_coarse[..., 0], axes=(1, 2)), model_coarse
      )[..., None]
      padded_J = periodic_pad(J[..., None])
      J_filter = conv(padded_J)
      # J_filter = jax.vmap(res_fn)(J[..., None])
      output = (J_filter - J_coarse) / model_coarse.dx**2
      x_hist = jax.vmap(res_fn)(model_fine.x_hist[..., None])
      index = 100
      x_next_coarse = model_coarse.CN_real(x_hist[index])
      x_hist[index + 1] - x_next_coarse + output[index] * model_coarse.dx**2 * dt
      breakpoint()
    elif sgs_model == "coarse_correction":
      w_coarse = jax.vmap(res_fn)(model_fine.x_hist[..., None])
      output = np.zeros_like(w_coarse)
      for k in range(model_fine.step_num):
        next_step_fine = model_fine.CN_real(model_fine.x_hist[k][..., None])
        next_step_coarse = model_coarse.CN_real(w_coarse[k])
        output[k] = (res_fn(next_step_fine) - next_step_coarse) / dt

    # output_ = np.zeros_like(w_coarse)
    # for k in range(model_fine.step_num):
    #   next_step_fine = model_fine.CN_real(model_fine.x_hist[k][..., None])
    #   next_step_coarse = model_coarse.CN_real(w_coarse[k])
    #   output_[k] = (res_fn(next_step_fine) - next_step_coarse) / dt

    if not output.shape[1] == output.shape[2] == n:
      breakpoint()
      raise Exception("The shape of output is wrong.")

    if not jnp.isnan(output).any() and not jnp.isinf(w_coarse).any():
      # successful generating traj
      u_[j] = w_coarse
      outputs[j] = output
      j = j + 1

  inputs = w_coarse
  n_plot = 4
  from matplotlib import cm
  from matplotlib import pyplot as plt
  fraction = 0.05
  pad = 0.001
  fig, axs = plt.subplots(2, n_plot, figsize=(12, 5))
  index_array = np.arange(
    0, n_plot * output.shape[0] // n_plot - 1, output.shape[0] // n_plot
  )
  for k in range(n_plot):
    im = axs[0, k].imshow(inputs[index_array[k]], cmap=cm.twilight)
    _ = fig.colorbar(
      im, ax=axs[0, k], orientation='horizontal', fraction=fraction, pad=pad
    )
    axs[0, k].axis("off")
    im = axs[1, k].imshow(output[index_array[k]], cmap=cm.twilight)
    _ = fig.colorbar(
      im, ax=axs[1, k], orientation='horizontal', fraction=fraction, pad=pad
    )
    axs[1, k].axis("off")
  fig.tight_layout(pad=0.0)
  plt.savefig("results/fig/dataset.png")
  plt.close()
  psi_hat = -what_hist / model.laplacian_[None]
  dpsidx = jnp.fft.irfft2(1j * psi_hat * model.kx[None], axes=(1, 2))
  dpsidy = jnp.fft.irfft2(1j * psi_hat * model.ky[None], axes=(1, 2))
  plt.plot(jnp.sum(inputs**2 + outputs**2, axis=(0, 2, 3)), label='e_kin')
  plt.plot(
    jnp.sum(inputs**2 + outputs**2, axis=(0, 2, 3))[0] *
    jnp.exp(-model_fine.nu * jnp.linspace(0, model_fine.T, int(T/dt)+1)), label='decay'
  )
  plt.legend()
  plt.savefig("results/fig/e_kin.png")

  breakpoint()
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
        u_.reshape(-1, n, n, 1),  # shape [case_num * step_num // writeInterval, nx, ny, 1]
        "outputs":
        outputs.reshape(-1, n, n, 1),  # shape [case_num * step_num // writeInterval, nx, ny, 1]
      },
      "config":
      config_dict,
      "readme":
      "This dataset contains the results of solving an incompressible"
      "Navier-Stokes equation using a pseudo-spectral method over a periodic"
      "2D box domain. "
      "The 'input' field represents the velocity, and the 'output' "
      "field represents the nonlinear RHS."
    }

    with h5py.File(
      f"data/ns_hit/Re{Re}_nx{n}_n{case_num}_{sgs_model}.h5", "w"
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
