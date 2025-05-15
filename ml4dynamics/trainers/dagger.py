"""
Implementation of the Dataset Aggregation (DAgger) algorithm

reference:
1. Ross, Stéphane, Geoffrey Gordon, and Drew Bagnell. "A reduction of imitation
learning and structured prediction to no-regret online learning." Proceeding
of the fourteenth international conference on artificial intelligence and
statistics. JMLR Workshop and Conference Proceedings, 2011.
"""

from functools import partial
from time import time

import jax
import jax.numpy as jnp
import jax.random as random
import ml_collections
import numpy as np
import torch
import yaml
from box import Box
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ml4dynamics.dataset_utils import dataset_utils
from ml4dynamics.trainers import train_utils
from ml4dynamics.utils import utils

jax.config.update("jax_enable_x64", True)


def main(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  # model parameters
  T = config.react_diff.T
  dt = config.react_diff.dt
  step_num = int(T / dt)
  nx = config.react_diff.nx
  r = config.react_diff.r
  # solver parameters
  dagger_epochs = config.dagger.epochs
  inner_epochs = config.dagger.inner_epochs
  beta = config.dagger.beta
  rng = random.PRNGKey(config.sim.seed)
  np.random.seed(rng)

  inputs, outputs, train_dataloader, test_dataloader = utils.load_data(
    config_dict, config.train.batch_size_unet, mode="jax"
  )
  train_state, schedule = utils.prepare_unet_train_state(config_dict)

  @partial(jax.jit, static_argnums=(3, ))
  def train_step(x, y, train_state, is_training=True):

    def loss_fn(params, batch_stats, is_training):
      y_pred, batch_stats = train_state.apply_fn_with_bn(
        {
          "params": params,
          "batch_stats": batch_stats
        },
        x,
        is_training=is_training
      )
      loss = jnp.mean((y - y_pred)**2)

      return loss, batch_stats

    if is_training:
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (loss, batch_stats
       ), grads = grad_fn(train_state.params, train_state.batch_stats, True)

      train_state = train_state.apply_gradients(grads=grads)
      train_state = train_state.update_batch_stats(batch_stats)
    else:
      loss, batch_stats = loss_fn(
        train_state.params, train_state.batch_stats, False
      )

    return loss, train_state

  rd_fine, rd_coarse = utils.create_fine_coarse_simulator(config)
  calc_correction = jax.jit(
    partial(dataset_utils.calc_correction, rd_fine, rd_coarse, nx, r)
  )
  # fix a case for DAgger iteration
  # key, rng = random.split(rng)
  # max_freq = 10
  # u_fft = jnp.zeros((nx, nx, 2))
  # u_fft = u_fft.at[:max_freq, :max_freq].set(
  #   random.normal(key, shape=(max_freq, max_freq, 2))
  # )
  # uv = jnp.real(jnp.fft.fftn(u_fft, axes=(0, 1))) / nx
  # uv = uv.transpose(2, 0, 1)

  dagger_iters = tqdm(range(dagger_epochs))
  for i in dagger_iters:
    print(f"DAgger {i}-th iteration...")
    print(f"{train_state.step}th step, lr: {schedule(train_state.step):.4e}")
    inner_iters = tqdm(range(inner_epochs))
    loss_hist = []
    for j in inner_iters:
      loss_avg = 0
      count = 1
      for batch_inputs, batch_outputs in train_dataloader:
        loss, train_state = train_step(
          jnp.array(batch_inputs), jnp.array(batch_outputs), train_state, False
        )
        loss_avg += loss
        count += 1
        desc_str = f"{loss=:.4e}"
        inner_iters.set_description_str(desc_str)
        loss_hist.append(loss)
      loss_avg /= count
      if jnp.isnan(loss):
        print("Training loss became NaN. Stopping training.")
        break

    plt.plot(jnp.array(loss_hist) - jnp.array(loss_hist).min() + 0.001)
    plt.yscale("log")
    plt.savefig(f"results/fig/{i}th_loss.pdf")
    plt.close()
    val_loss = 0
    count = 0
    for batch_inputs, batch_outputs in test_dataloader:
      loss, train_state = train_step(
        jnp.array(batch_inputs), jnp.array(batch_outputs), train_state, False
      )
      val_loss += loss
      count += 1
    print(f"val loss: {val_loss/count:0.4f}")

    # DAgger step
    key, rng = random.split(rng)
    max_freq = 10
    u_fft = jnp.zeros((nx, nx, 2))
    u_fft = u_fft.at[:max_freq, :max_freq].set(
      random.normal(key, shape=(max_freq, max_freq, 2))
    )
    uv = jnp.real(jnp.fft.fftn(u_fft, axes=(0, 1))) / nx
    uv = uv.transpose(2, 0, 1)
    start = time()
    x_hist = train_utils.run_simulation_coarse_grid_correction(
      train_state, rd_coarse, outputs, nx, r, dt, beta, uv
    )
    print(f"simulation takes {time() - start:.2f}s...")
    if jnp.any(jnp.isnan(x_hist)) or jnp.any(jnp.isinf(x_hist)):
      print("similation contains NaN!")
      breakpoint()
    rd_fine.run_simulation(uv.reshape(-1), rd_fine.adi)
    print(
      "L2 error: {:.4f}".format(
        np.sum(
          np.linalg.norm(x_hist.reshape(step_num, -1) - rd_fine.x_hist, axis=1)
        )
      )
    )
    input = x_hist.reshape((step_num, 2, nx, nx))
    output = jnp.zeros_like(input)
    for j in range(x_hist.shape[0]):
      output = output.at[j].set(calc_correction(input[j]) / dt)

    # visualization
    n_plot = 6
    fig, axs = plt.subplots(3, n_plot, figsize=(12, 6))
    for j in range(n_plot):
      im = axs[0, j].imshow(
        rd_fine.x_hist[j * 500, :nx**2].reshape(nx, nx), cmap=cm.twilight
      )
      _ = fig.colorbar(im, ax=axs[0, j], orientation='horizontal')
      axs[0, j].axis("off")
      im = axs[1, j].imshow(x_hist[j * 500, 0], cmap=cm.twilight)
      _ = fig.colorbar(im, ax=axs[1, j], orientation='horizontal')
      axs[1, j].axis("off")
      im = axs[2, j].imshow(
        rd_fine.x_hist[j * 500, :nx**2].reshape(nx, nx) - x_hist[j * 500, 0],
        cmap=cm.twilight
      )
      _ = fig.colorbar(im, ax=axs[2, j], orientation='horizontal')
      axs[2, j].axis("off")

    plt.savefig(f"results/fig/dagger_cloudmap_{i}.pdf")
    plt.close()

    # generate new dataset
    inputs = np.vstack([inputs, np.asarray(input.transpose(0, 2, 3, 1))])
    outputs = np.vstack([outputs, np.asarray(output.transpose(0, 2, 3, 1))])
    train_x, test_x, train_y, test_y = train_test_split(
      inputs, outputs, test_size=0.2, random_state=config.sim.seed
    )
    train_dataset = TensorDataset(
      torch.tensor(train_x, dtype=torch.float64),
      torch.tensor(train_y, dtype=torch.float64)
    )
    train_dataloader = DataLoader(
      train_dataset,
      batch_size=config.train.batch_size_unet,
      shuffle=True  # , num_workers=num_workers
    )
    test_dataset = TensorDataset(
      torch.tensor(test_x, dtype=torch.float64),
      torch.tensor(test_y, dtype=torch.float64)
    )
    test_dataloader = DataLoader(
      test_dataset,
      batch_size=config.train.batch_size_unet,
      shuffle=True  # , num_workers=num_workers
    )


if __name__ == "__main__":

  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
