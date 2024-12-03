import os

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import torch
import yaml
from box import Box
from dlpack import asdlpack
from jax import random
from matplotlib import cm
from matplotlib import pyplot as plt
from ml4dynamics.dynamics import RD
from ml4dynamics.models.models import UNet
from time import time

jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=15)
torch.set_default_dtype(torch.float64)


def a_posteriori_test(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  pde_type = config.name
  alpha = config.react_diff.alpha
  beta = config.react_diff.beta
  gamma = config.react_diff.gamma
  d = config.react_diff.d
  T = config.react_diff.T
  dt = config.react_diff.dt
  step_num = int(T / dt)
  Lx = config.react_diff.Lx
  nx = config.react_diff.nx
  r = config.react_diff.r
  case_num = config.sim.case_num
  # rng = np.random.PRNGKey(config.sim.seed)
  dataset = "alpha{:.2f}_beta{:.2f}_gamma{:.2f}_n{}".format(
    alpha, beta, gamma, case_num
  )
  if pde_type == "react_diff":
    h5_filename = f"data/react_diff/{dataset}.h5"
  GPU = 0
  device = torch.device(
    "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"
  )
  model = UNet().to(device)
  if os.path.isfile("ckpts/{}/ols-{}.pth".format(pde_type, dataset)):
      model.load_state_dict(
        torch.load(
          "ckpts/{}/ols-{}.pth".format(pde_type, dataset),
          map_location=torch.device("cpu")
        )
      )
      model.eval()

  rd_fine = RD(
    L=Lx,
    N=nx**2 * 2,
    T=T,
    dt=dt,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    d=d,
  )
  rd_coarse = RD(
    L=Lx,
    N=(nx//r)**2 * 2,
    T=T,
    dt=dt,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    d=d,
  )

  def run_simulation(uv: np.ndarray):
    step_num = rd_fine.step_num
    x_hist = jnp.zeros([step_num, 2, nx, nx])
    for i in range(step_num):
      x_hist = x_hist.at[i].set(uv)
      tmp = (uv[:, 0::2, 0::2] + uv[:, 1::2, 0::2] +
        uv[:, 0::2, 1::2] + uv[:, 1::2, 1::2]) / 4
      uv = rd_coarse.adi(tmp.reshape(-1)).reshape(2, nx // r, nx // r)
      uv = np.vstack(
        [np.kron(uv[0], np.ones((r, r))).reshape(1, nx, nx),
        np.kron(uv[1], np.ones((r, r))).reshape(1, nx, nx),]
      )

      uv_np = np.asarray(uv)
      uv_torch = torch.from_numpy(uv_np).clone().to(device)
      correction = model(uv_torch.reshape((1, *uv.shape)))
      uv += jnp.array((correction[0].detach().cpu().numpy()))
      # uv_torch = torch.tensor(uv, dtype=torch.float64).clone().to(device)
      # correction = model(
      #   torch.from_dlpack(asdlpack(uv)).reshape((1, *uv.shape))
      # )
      # uv += jnp.array(jax.numpy.from_dlpack(asdlpack(correction[0].detach())))

    return x_hist
  
  u_fft = jnp.zeros((2, nx, nx))
  u_fft = u_fft.at[:, :10, :10].set(
    random.normal(key=random.PRNGKey(0), shape=(2, 10, 10))
  )
  uv0 = jnp.real(jnp.fft.fftn(u_fft, axes=(1, 2))) / nx
  start = time()
  x_hist = run_simulation(uv0)
  print(f"running correction simulation takes {time() - start:.4f}...")
  n_plot = 3
  fig, axs = plt.subplots(n_plot, n_plot)
  axs = axs.flatten()
  for i in range(n_plot**2):
    axs[i].imshow(x_hist[i * 500, 0], cmap=cm.jet)
    axs[i].set_title(f"{x_hist[i * 500, 0].max():.4f}")
    axs[i].axis("off")
  plt.savefig("results/fig/rd.pdf")

if __name__ == "__main__":
   with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
   a_posteriori_test(config_dict)