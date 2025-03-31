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
from tqdm import tqdm

from ml4dynamics import utils
from ml4dynamics.models.models_jax import CustomTrainState, UNet
from ml4dynamics.trainers import train_utils
from ml4dynamics.types import PRNGKey

jax.config.update("jax_enable_x64", True)

def main(config: ml_collections.ConfigDict):

  def train(
    model_type: str,
    epochs: int,
    rng: PRNGKey,
  ):

    @partial(jax.jit, static_argnums=(3, 4))
    def train_step(x, y, train_state, loss_fn, is_training=True):
       
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

        if model_type == "ae":
          loss = jnp.mean((x - y_pred)**2)
        elif model_type == "mols":
          squared_norms = jax.tree_map(lambda param: jnp.sum(param ** 2), params)
          loss += jax.tree_util.tree_reduce(lambda x, y: x + y, squared_norms) * config.tr.lambda_mols
        elif model_type == "aols":
          pass
        elif model_type == "tr":
          pass

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

    unet = UNet()
    rng1, rng2 = random.split(rng)
    init_rngs = {
      'params': rng1,
      'dropout': rng2
    }
    unet_variables = unet.init(init_rngs, jnp.ones([1, nx, nx, 2]))
    schedule = optax.piecewise_constant_schedule(
      init_value=config.train.lr, boundaries_and_scales={
        int(b): 0.5 for b in jnp.arange(1, config.train.epochs, 100) 
      }
    )
    optimizer = optax.adam(schedule)
    train_state = CustomTrainState.create(
      apply_fn=unet.apply,
      params=unet_variables["params"],
      tx=optimizer,
      batch_stats=unet_variables["batch_stats"]
    )
    loss_hist = []

    iters = tqdm(range(epochs))
    for epoch in iters:
      total_loss = 0
      for k in range(0, train_x.shape[0], batch_size):
        loss, train_state = train_step(
          train_x[k: k + batch_size],
          train_y[k: k + batch_size],
          train_state,
          True
        )
        total_loss += loss
      desc_str = f"{total_loss=:.4f}"
      iters.set_description_str(desc_str)
      loss_hist.append(total_loss)
      
      if (epoch + 1) % config.train.save == 0:
        with open(f"ckpts/{pde_type}/{model_type}.pkl", "wb") as f:
          pickle.dump(unet_variables, f)

    val_loss = 0
    for k in range(0, test_x.shape[0], batch_size):
      loss, train_state = train_step(
        jnp.array(test_x[k: k + batch_size]),
        jnp.array(test_y[k: k + batch_size]),
        train_state,
        False
      )
      val_loss += loss
    print(f"val loss: {val_loss:.4f}")
    beta = 0
    run_simulation = partial(
      train_utils.run_simulation_coarse_grid_correction, train_state, rd_fine,
      rd_coarse, nx, r, dt, beta
    )
    start = time()
    x_hist = run_simulation(train_x[0].transpose(2, 0, 1))
    rd_fine.run_simulation(
      (train_x[0].transpose(2, 0, 1)).reshape(-1), rd_fine.adi
    )
    step_num = rd_fine.step_num
    print(f"simulation takes {time() - start:.2f}s...")
    if jnp.any(jnp.isnan(x_hist)) or jnp.any(jnp.isinf(x_hist)):
      print("similation contains NaN!")
      breakpoint()
    print("L2 error: {:.4f}".format(
      jnp.sum(
        jnp.linalg.norm(x_hist.reshape(step_num, -1) - rd_fine.x_hist, axis=1)
      )
    ))

    # visualization
    n_plot = 6
    fig, axs = plt.subplots(3, n_plot, figsize=(12, 6))
    for j in range(n_plot):
      im = axs[0, j].imshow(
        rd_fine.x_hist[j * 500, :nx**2].reshape(nx, nx), cmap=cm.jet
      )
      cbar = fig.colorbar(im, ax=axs[0, j], orientation='horizontal')
      axs[0, j].axis("off")
      im = axs[1, j].imshow(
        x_hist[j * 500, 0], cmap=cm.jet
      )
      cbar = fig.colorbar(im, ax=axs[1, j], orientation='horizontal')
      axs[1, j].axis("off")
      im = axs[2, j].imshow(
        rd_fine.x_hist[j * 500, :nx**2].reshape(nx, nx) - x_hist[j * 500, 0],
        cmap=cm.jet
      )
      cbar = fig.colorbar(im, ax=axs[2, j], orientation='horizontal')
      axs[2, j].axis("off")
      
    plt.savefig(f"results/fig/tr_cloudmap.pdf")
    plt.clf()

  config = Box(config_dict)
  pde_type = config.name
  nx = config.react_diff.nx
  r = config.react_diff.r
  dt = config.react_diff.dt
  batch_size = config.train.batch_size_jax
  epochs = config.train.epochs

  print("start loading data...")
  start = time()
  inputs, outputs, train_x, test_x, train_y, test_y = utils.load_data(config)
  print(f"finis loading data with {time() - start:.2f}s...")
  rd_fine, rd_coarse = utils.create_fine_coarse_simulator(config)

  rng = jax.random.PRNGKey(config.sim.seed)
  models_array = ["ae", "ols", "mols", "aols", "tr"]

  for _ in models_array:
    rng, key = random.split(rng)
    print(f"Training {_}...")
    train(_, epochs, key)

if __name__ == "__main__":

  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)