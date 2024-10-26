from functools import partial
from typing import Iterator, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import yaml
from box import Box
from jax import random as random
from jaxtyping import Array
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dynamics import KS
from src.types import Batch, OptState, PRNGKey
from src.utils import plot_with_horizontal_colorbar


def main(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  # model parameters
  nu1 = config.ks.nu
  c = config.ks.c
  L = config.ks.L
  T = config.ks.T
  init_scale = config.ks.init_scale
  # solver parameters
  N1 = config.ks.nx
  N2 = N1 // 2
  dt = config.ks.dt
  r = N1 // N2
  key = random.PRNGKey(config.sim.seed)

  # fine simulation
  ks_fine = KS(
    N=N1,
    T=T,
    dt=dt,
    dx=L / (N1 + 1),
    tol=1e-8,
    init_scale=init_scale,
    tv_scale=1e-8,
    L=L,
    nu=nu1,
    c=c,
    key=key,
  )
  # coarse simulator
  ks_coarse = KS(
    N=N2,
    T=T,
    dt=dt,
    dx=L / (N2 + 1),
    tol=1e-8,
    init_scale=init_scale,
    tv_scale=1e-8,
    L=L,
    nu=nu1,
    c=c,
    key=key,
  )

  # define the restriction and interpolation operator
  # TODO: try to change the restriction and projection operator to test the
  # results, these operator should have test file
  res_op = jnp.zeros((N2, N1))
  int_op = jnp.zeros((N1, N2))
  res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r].set(1)
  int_op = int_op.at[jnp.arange(N2) * r, jnp.arange(N2)].set(1)

  assert jnp.allclose(res_op @ int_op, jnp.eye(N2))

  input = jnp.zeros((config.sim.case_num, int(T / dt), N2))
  output = jnp.zeros((config.sim.case_num, int(T / dt), N2))
  for i in range(config.sim.case_num):
    key, subkey = random.split(key)
    x = ks_fine.attractor + init_scale * random.normal(
      subkey, shape=(ks_fine.N, )
    )
    # ks_fine.run_simulation(x, ks_fine.CN_FEM)
    ks_fine.run_simulation(ks_fine.x_targethist[0], ks_fine.CN_FEM)
    input_ = ks_fine.x_hist @ res_op.T  # shape = [step_num, N2]
    output_ = jnp.zeros_like(input_)
    for j in range(ks_fine.step_num):
      next_step_fine = ks_fine.CN_FEM(
        ks_fine.x_hist[j]
      )  # shape = [N1, step_num]
      next_step_coarse = ks_coarse.CN_FEM(input_[j])  # shape = [step_num, N2]
      output_ = output_.at[j].set(res_op @ next_step_fine - next_step_coarse)

    input = input.at[i].set(input_)
    output = output.at[i].set(output_)

  input = input.reshape(-1, N2)
  output = output.reshape(-1, N2)
  if jnp.any(jnp.isnan(input)) or jnp.any(jnp.isnan(output)) or\
    jnp.any(jnp.isinf(input)) or jnp.any(jnp.isinf(output)):
    raise Exception("The data contains Inf or NaN")
  np.savez('data/ks/tmp.npz', input=input, output=output)

  # train test split
  train_x, test_x, train_y, test_y = train_test_split(
    input, output, test_size=0.2, random_state=42
  )
  train_ds = {"input": jnp.array(train_x), "output": jnp.array(train_y)}
  test_ds = {"input": jnp.array(test_x), "output": jnp.array(test_y)}

  # training a fully connected neural network to do the closure modeling
  def sgs_fn(features: jnp.ndarray) -> jnp.ndarray:
    """
    NOTE: an example to show the inconsistency of a priori and a posteriori
    error. Fix the network architecture to be the same, compare the results
    where the lr is chosen to be 1e-3 and 1e-4.
    """

    mlp = hk.Sequential(
      [
        hk.Flatten(),
        hk.Linear(2048),
        jax.nn.relu,
        hk.Linear(1024),
        jax.nn.relu,
        hk.Linear(N2),
      ]
    )
    linear_residue = hk.Linear(N2)
    return mlp(features)  # + linear_residue(features)

  correction_nn = hk.without_apply_rng(hk.transform(sgs_fn))
  # 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-5
  lr = 5e-4
  optimizer = optax.adam(lr)
  params = correction_nn.init(random.PRNGKey(0), np.zeros((1, N2)))
  opt_state = optimizer.init(params)

  @jax.jit
  def loss_fn(
    params: hk.Params, input: jnp.ndarray, output: jnp.ndarray
  ) -> float:
    predict = correction_nn.apply(params, input)
    return jnp.mean((output - predict)**2)

  @jax.jit
  def update(
    params: hk.Params, input: jnp.ndarray, output: jnp.ndarray,
    opt_state: OptState
  ) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(
      partial(loss_fn, input=input, output=output)
    )(params)
    # loss, grads = jax.value_and_grad(loss_fn)(params, input, output)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

  loss_hist = []
  epochs = 10000
  batch_size = 1000
  iters = tqdm(range(epochs))
  for step in iters:
    for i in range(0, len(train_ds["input"]), batch_size):
      input = train_ds["input"][i:i + batch_size]
      output = train_ds["output"][i:i + batch_size]
      loss, params, opt_state = update(params, input, output, opt_state)
      loss_hist.append(loss)
      relative_loss = loss / jnp.mean(output**2)
      desc_str = f"{relative_loss=:.4e}"
      iters.set_description_str(desc_str)

  breakpoint()
  valid_loss = 0
  for i in range(0, len(test_ds["input"]), batch_size):
    input = train_ds["input"][i:i + batch_size]
    output = train_ds["output"][i:i + batch_size]
    valid_loss += loss_fn(params, input=input, output=output)
  print("lr: {:.2e}".format(lr))
  print("validation loss: {:.4e}".format(valid_loss))

  # a posteriori error estimate
  key, subkey = random.split(key)
  x = ks_fine.attractor + init_scale * random.normal(
    subkey, shape=(ks_fine.N, )
  )
  ks_fine.run_simulation(x, ks_fine.CN_FEM)
  ks_coarse.run_simulation(x[::r], ks_coarse.CN_FEM)
  im_array = jnp.zeros(
    (3, 1, ks_coarse.x_hist.shape[1], ks_coarse.x_hist.shape[0])
  )
  im_array = im_array.at[0, 0].set(ks_fine.x_hist[:, ::r].T)
  im_array = im_array.at[1, 0].set(ks_coarse.x_hist.T)
  im_array = im_array.at[2,
                         0].set(ks_coarse.x_hist.T - ks_fine.x_hist[:, ::r].T)
  title_array = [f"{N1}", f"{N2}", "diff"]
  plot_with_horizontal_colorbar(
    im_array,
    fig_size=(4, 6),
    title_array=title_array,
    file_path=f"results/fig/ks_nu{nu1}_N1{N1}N2{N2}_cmp.pdf"
  )
  print(
    "rmse without correction: {:.4f}".format(
      jnp.sqrt(jnp.mean((ks_coarse.x_hist.T - ks_fine.x_hist[:, ::r].T)**2))
    )
  )

  corrector = partial(correction_nn.apply, params)
  ks_coarse.run_simulation_with_correction(x[::r], ks_coarse.CN_FEM, corrector)
  im_array = jnp.zeros(
    (3, 1, ks_coarse.x_hist.shape[1], ks_coarse.x_hist.shape[0])
  )
  im_array = im_array.at[0, 0].set(ks_fine.x_hist[:, ::r].T)
  im_array = im_array.at[1, 0].set(ks_coarse.x_hist.T)
  im_array = im_array.at[2,
                         0].set(ks_coarse.x_hist.T - ks_fine.x_hist[:, ::r].T)
  plot_with_horizontal_colorbar(
    im_array,
    fig_size=(4, 6),
    title_array=title_array,
    file_path=f"results/fig/ks_nu{nu1}_N1{N1}N2{N2}_correct_cmp.pdf"
  )
  print(
    "rmse with correction: {:.4f}".format(
      jnp.sqrt(jnp.mean((ks_coarse.x_hist.T - ks_fine.x_hist[:, ::r].T)**2))
    )
  )


if __name__ == "__main__":

  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
