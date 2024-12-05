"""
Implementation of the Dataset Aggregation (DAgger) algorithm

reference:
1. Ross, Stéphane, Geoffrey Gordon, and Drew Bagnell. "A reduction of imitation
learning and structured prediction to no-regret online learning." Proceeding
of the fourteenth international conference on artificial intelligence and
statistics. JMLR Workshop and Conference Proceedings, 2011.
"""
import os
from time import time

import h5py
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
import yaml
from box import Box
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ml4dynamics.models.models_jax import UNet

np.set_printoptions(precision=15)
jax.config.update("jax_enable_x64", True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
CKPT_DIR = 'ckpts'


def main(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  # model parameters
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
  dx = Lx / nx
  r = config.react_diff.r
  # solver parameters
  dagger_epochs = config.train.dagger_epochs
  epochs = config.train.epochs_ae
  batch_size = config.train.batch_size_ae
  # rng = random.PRNGKey(config.sim.seed)
  case_num = config.sim.case_num

  dataset = "alpha{:.2f}_beta{:.2f}_gamma{:.2f}_n{}".format(
    alpha, beta, gamma, case_num
  )
  if pde_type == "react_diff":
    h5_filename = f"data/react_diff/{dataset}.h5"

  with h5py.File(h5_filename, "r") as h5f:
    inputs = jnp.array(h5f["data"]["inputs"][()]).transpose(0, 2, 3, 1)
    outputs = jnp.array(h5f["data"]["inputs"][()]).transpose(0, 2, 3, 1)

  print(f"Training {pde_type} model with data: {dataset} ...")
  train_x, test_x, train_y, test_y = train_test_split(
    inputs, outputs, test_size=0.2, random_state=config.sim.seed
  )
  dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
  dataset = dataset.shuffle(buffer_size=4).batch(2)

  unet = UNet()
  init_rngs = {
    'params': jax.random.PRNGKey(0),
    'dropout': jax.random.PRNGKey(1)
  }
  unet_variables = unet.init(init_rngs, jnp.ones([1, nx, nx, 2]))
  optimizer = optax.adam(learning_rate=0.001)
  opt_state = optimizer.init(unet_variables)
  # train_state = CustomTrainState.create(
  #   apply_fn=unet.apply, params=unet_variables["params"], tx=optimizer,
  #   batch_stats=unet_variables["batch_stats"]
  # )

  @jax.jit
  def update(variables, x, y, opt_state):

    def loss_fn(params, batch_stats):
      predict = unet.apply({"params": params, "batch_stats": batch_stats}, x)
      return jnp.mean((y - predict))

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, batch_stats
     ), grads = grad_fn(variables["params"], variables["batch_stats"])
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  iters = tqdm(range(dagger_epochs))
  for e in iters:
    loss_avg = 0
    count = 1
    tic = time()
    for batch_data, batch_labels in dataset:
      breakpoint()
      loss, unet_variables, opt_state = update(
        unet_variables, jnp.array(batch_data), jnp.array(batch_labels),
        opt_state
      )
      loss_avg += loss
      count += 1
      desc_str = f"{loss=:.4e}"
      iters.set_description_str(desc_str)

    loss_avg /= count
    elapsed = time() - tic
    print(f"epoch: {e}, loss: {loss_avg:0.2f}, elapased: {elapsed:0.2f}")

    if np.isnan(loss):
      print("Training loss became NaN. Stopping training.")
      break

  # checkpoints.save_checkpoint(
  #   ckpt_dir=CKPT_DIR, target=train_state, step=0, overwrite=True
  # )


if __name__ == "__main__":

  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
