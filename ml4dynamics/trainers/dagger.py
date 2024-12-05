"""
Implementation of the Dataset Aggregation (DAgger) algorithm

reference:
1. Ross, Stéphane, Geoffrey Gordon, and Drew Bagnell. "A reduction of imitation
learning and structured prediction to no-regret online learning." Proceeding
of the fourteenth international conference on artificial intelligence and
statistics. JMLR Workshop and Conference Proceedings, 2011.
"""

import h5py
import jax
import jax.numpy as jnp
import jax.random as random
import ml_collections
import optax
import tensorflow as tf
import yaml
from box import Box
from functools import partial
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ml4dynamics.dynamics import RD
from ml4dynamics.models.models_jax import UNet, CustomTrainState

jax.config.update("jax_enable_x64", True)


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
  epochs = config.train.dagger_inner_epochs
  batch_size = config.train.batch_size_ae
  rng = random.PRNGKey(config.sim.seed)
  case_num = config.sim.case_num

  dataset = "alpha{:.2f}_beta{:.2f}_gamma{:.2f}_n{}".format(
    alpha, beta, gamma, case_num
  )
  if pde_type == "react_diff":
    h5_filename = f"data/react_diff/{dataset}.h5"

  with h5py.File(h5_filename, "r") as h5f:
    inputs = jnp.array(h5f["data"]["inputs"][()]).transpose(0, 2, 3, 1)
    outputs = jnp.array(h5f["data"]["inputs"][()]).transpose(0, 2, 3, 1)
  print(
    f"Training {pde_type} model with data: {dataset} ..."
  )
  train_x, test_x, train_y, test_y = train_test_split(
    inputs, outputs, test_size=0.2, random_state=config.sim.seed
  )
  dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
  dataset = dataset.shuffle(buffer_size=4).batch(200)
  val_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
  val_dataset = val_dataset.shuffle(buffer_size=4).batch(200)
  
  unet = UNet()
  init_rngs = {
    'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(1)
  }
  unet_variables = unet.init(init_rngs, jnp.ones([1, nx, nx, 2]))
  optimizer = optax.adam(learning_rate=0.01)
  train_state = CustomTrainState.create(
    apply_fn=unet.apply, params=unet_variables["params"], tx=optimizer,
    batch_stats=unet_variables["batch_stats"]
  )

  @partial(jax.jit, static_argnums=(3,))
  def train_step(x, y, train_state, is_training=True):
    def loss_fn(params, batch_stats, is_training):
      y_pred, batch_stats = train_state.apply_fn_with_bn(
        {"params": params, "batch_stats": batch_stats}, x,
        is_training=is_training
      )
      loss = jnp.mean((y - y_pred)**2)

      return loss, batch_stats

    if is_training:
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (loss, batch_stats), grads = grad_fn(
        train_state.params, train_state.batch_stats, True
      )

      train_state = train_state.apply_gradients(grads=grads)
      train_state = train_state.update_batch_stats(batch_stats)
    else:
      loss, batch_stats = loss_fn(
        train_state.params, train_state.batch_stats, False
      )

    return loss, train_state

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

  @jax.jit
  def run_simulation(uv: jnp.ndarray):
    step_num = rd_fine.step_num
    x_hist = jnp.zeros([step_num, 2, nx, nx])
    for i in range(step_num):
      x_hist = x_hist.at[i].set(uv)
      uv = uv.transpose(1, 2, 0)
      correction, _ = train_state.apply_fn_with_bn(
        {"params": train_state.params, "batch_stats": train_state.batch_stats},
        uv.reshape(1, *uv.shape), is_training=False
      )
      correction = correction.reshape(nx, nx, -1).transpose(2, 0, 1)
      uv = uv.transpose(2, 0, 1)
      tmp = (uv[:, 0::2, 0::2] + uv[:, 1::2, 0::2] +
        uv[:, 0::2, 1::2] + uv[:, 1::2, 1::2]) / 4
      uv = rd_coarse.adi(tmp.reshape(-1)).reshape(2, nx // r, nx // r)
      uv = jnp.vstack(
        [jnp.kron(uv[0], jnp.ones((r, r))).reshape(1, nx, nx),
         jnp.kron(uv[1], jnp.ones((r, r))).reshape(1, nx, nx),]
      )
      uv += correction * dt

    return x_hist

  dagger_iters = tqdm(range(dagger_epochs))
  for i in dagger_iters:
    print(f"DAgger {i}-th iteration")
    inner_iters = tqdm(range(epochs))
    for e in inner_iters:
      loss_avg = 0
      count = 1
      for batch_data, batch_labels in dataset:
        loss, train_state = train_step(
          jnp.array(batch_data), jnp.array(batch_labels), train_state, True
        )
        loss_avg += loss
        count += 1
        desc_str = f"{loss=:.4f}"
        inner_iters.set_description_str(desc_str)
      loss_avg /= count
      if jnp.isnan(loss):
        print("Training loss became NaN. Stopping training.")
        break
    
    val_loss = 0
    count = 0
    for batch_data, batch_labels in val_dataset:
      loss, train_state = train_step(
        jnp.array(batch_data), jnp.array(batch_labels), train_state, False
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
    x_hist = run_simulation(uv)
    if jnp.any(jnp.isnan(x_hist)) or jnp.any(jnp.isinf(x_hist)):
      print("similation contains NaN!")
      breakpoint()
    input = rd_fine.x_hist.reshape((step_num, 2, nx, nx))
    output = jnp.zeros_like(input)
    for j in range(x_hist.shape[0]):
      next_step_fine = rd_fine.adi(x_hist[i]).reshape(2, nx, nx)
      tmp = (input[i, :, 0::2, 0::2] + input[i, :, 1::2, 0::2] +
        input[i, :, 0::2, 1::2] + input[i, :, 1::2, 1::2]) / 4
      uv = tmp.reshape(-1)
      next_step_coarse = rd_coarse.adi(uv).reshape(2, nx // r, nx // r)
      next_steo_coarse_interp = jnp.vstack(
        [jnp.kron(next_step_coarse[0], jnp.ones((r, r))).reshape(1, nx, nx),
         jnp.kron(next_step_coarse[1], jnp.ones((r, r))).reshape(1, nx, nx),]
      )
      output = output.at[j].set(next_step_fine - next_steo_coarse_interp)

    # generate new dataset
    inputs = jnp.vstack([inputs, input])
    outputs = jnp.vstack([outputs, output])
    train_x, test_x, train_y, test_y = train_test_split(
      inputs, outputs, test_size=0.2, random_state=config.sim.seed
    )
    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    dataset = dataset.shuffle(buffer_size=4).batch(200)
    val_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    val_dataset = val_dataset.shuffle(buffer_size=4).batch(200)

if __name__ == "__main__":

  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
