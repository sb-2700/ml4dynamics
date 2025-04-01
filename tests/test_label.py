from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
import ml_collections
import numpy as np
import optax
import pickle
from time import time
import yaml
from box import Box
from matplotlib import cm
from matplotlib import pyplot as plt

from ml4dynamics import utils
from ml4dynamics.models.models_jax import CustomTrainState, UNet
from ml4dynamics.trainers import train_utils
from ml4dynamics.types import PRNGKey

jax.config.update("jax_enable_x64", True)

with open("config/simulation.yaml", "r") as file:
  config_dict = yaml.safe_load(file)
config = Box(config_dict)
nx = config.react_diff.nx
r = config.react_diff.r
dt = config.react_diff.dt
rng = jax.random.PRNGKey(0)
plot_ = False
unet = UNet()
rng1, rng2 = random.split(rng)
init_rngs = {
  'params': rng1,
  'dropout': rng2
}
unet_variables = unet.init(init_rngs, jnp.ones([1, nx, nx, 2]))
optimizer = optax.adam(config.train.lr)
train_state = CustomTrainState.create(
  apply_fn=unet.apply,
  params=unet_variables["params"],
  tx=optimizer,
  batch_stats=unet_variables["batch_stats"]
)
inputs, outputs, train_x, test_x, train_y, test_y = utils.load_data(config)

_, rd_coarse = utils.create_fine_coarse_simulator(config)
beta = 1
x_hist = train_utils.run_simulation_coarse_grid_correction(
  train_state, rd_coarse, outputs, nx, r, dt, beta,
  inputs[0].transpose(2, 0, 1)
)

step_num = 5000
err = jnp.linalg.norm(x_hist.reshape(step_num, -1)
  - jnp.transpose(inputs, (0, 3, 1, 2)).reshape(step_num, -1), axis=1)
print("L2 error: {:.4f}".format(jnp.sum(err)))
# visualization
if plot_:
  n_plot = 6
  fig, axs = plt.subplots(3, n_plot, figsize=(12, 6))
  for j in range(n_plot):
    im = axs[0, j].imshow(
      inputs[j * 500, ..., 0].reshape(nx, nx), cmap=cm.jet
    )
    cbar = fig.colorbar(im, ax=axs[0, j], orientation='horizontal')
    axs[0, j].axis("off")
    im = axs[1, j].imshow(
      x_hist[j * 500, 0], cmap=cm.jet
    )
    cbar = fig.colorbar(im, ax=axs[1, j], orientation='horizontal')
    axs[1, j].axis("off")
    im = axs[2, j].imshow(
      inputs[j * 500, ..., 0].reshape(nx, nx) - x_hist[j * 500, 0],
      cmap=cm.jet
    )
    cbar = fig.colorbar(im, ax=axs[2, j], orientation='horizontal')
    axs[2, j].axis("off")

  plt.savefig(f"results/fig/test_cloudmap.pdf")
  plt.clf()
  plt.plot(err)
  plt.yscale("log")
  plt.savefig("results/fig/err.pdf")
