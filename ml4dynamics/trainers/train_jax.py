from functools import partial

import h5py
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ml4dynamics.dataset_utils import dataset_utils
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

        if model_type == "mols":
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

  config = Box(config_dict)
  pde_type = config.name
  alpha = config.react_diff.alpha
  beta = config.react_diff.beta
  gamma = config.react_diff.gamma
  nx = config.react_diff.nx
  case_num = config.sim.case_num
  batch_size = config.train.batch_size_jax
  epochs = config.train.epochs

  # load dataset
  dataset = "alpha{:.2f}_beta{:.2f}_gamma{:.2f}_n{}".format(
    alpha, beta, gamma, case_num
  )
  if pde_type == "react_diff":
    h5_filename = f"data/react_diff/{dataset}.h5"

  print("start loading data...")
  start = time()
  with h5py.File(h5_filename, "r") as h5f:
    inputs = np.array(h5f["data"]["inputs"][()]).transpose(0, 2, 3, 1)
    outputs = np.array(h5f["data"]["inputs"][()]).transpose(0, 2, 3, 1)
  train_x, test_x, train_y, test_y = train_test_split(
    inputs, outputs, test_size=0.2, random_state=config.sim.seed
  )
  datasize = train_x.shape[0]
  shuffled_indices = np.random.permutation(datasize)
  train_x = train_x[shuffled_indices]
  train_y = train_y[shuffled_indices]
  print(f"finis loading data with {time() - start:.2f}s...")

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