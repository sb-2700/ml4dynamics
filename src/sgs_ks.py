from typing import Iterator, Optional, Tuple
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
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

def main():

  # model parameters
  nu1 = .1
  c = .1
  L = 10 * jnp.pi
  T = 40
  init_scale = 1.
  # solver parameters
  N1 = 512
  N2 = 256
  dt = 0.01
  r = N1 // N2
  key = random.PRNGKey(42)

  # fine simulation
  ks_fine = KS(  
    N = N1, T = T, dt = dt, dx = L / (N1+1), tol = 1e-8,
    init_scale = init_scale, tv_scale = 1e-8, L = L, nu = nu1, c = c, key=key,
  )
  ks_fine.run_simulation(ks_fine.x_targethist[0], ks_fine.CN_FEM)

  # coarse simulation
  ks_coarse = KS(
    N = N2, T = T, dt = dt, dx = L / (N2 + 1), tol = 1e-8,
    init_scale = init_scale, tv_scale = 1e-8, L = L, nu = nu1, c = c, key=key,
  )
  ks_coarse.run_simulation(ks_fine.x_hist[0, ::r], ks_coarse.CN_FEM)

  # define the restriction and interpolation operator
  res_op = jnp.zeros((N2, N1))
  int_op = jnp.zeros((N1, N2))
  res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r].set(1)
  int_op = int_op.at[jnp.arange(N2) * r, jnp.arange(N2)].set(1)

  assert jnp.allclose(res_op @ int_op, jnp.eye(N2))

  input = ks_fine.x_hist @ res_op.T # shape = [step_num, N2]
  output = jnp.zeros_like(input)
  for i in range(ks_fine.step_num):
    next_step_fine = ks_fine.CN_FEM(ks_fine.x_hist[i]) # shape = [N1, step_num]
    next_step_coarse = ks_coarse.CN_FEM(input[i]) # shape = [step_num, N2]
    output = output.at[i].set(res_op @ next_step_fine - next_step_coarse)

  np.savez('data/ks/tmp.npz', input = input, output = output)

  # train test split
  train_x, test_x, train_y, test_y = train_test_split(input, output, test_size=0.2, random_state=42)
  train_ds = {
    "input": jnp.array(train_x),
    "output": jnp.array(train_y)
  }
  test_ds = {
    "input": jnp.array(test_x),
    "output": jnp.array(test_y)
  }

  # training a fully connected neural network to do the closure modeling
  def sgs_fn(features: jnp.ndarray) -> jnp.ndarray:

    mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(2048), jax.nn.relu,
      hk.Linear(1024), jax.nn.relu,
      hk.Linear(N2), jax.nn.sigmoid
    ])
    return mlp(features)
  
  correction_nn = hk.without_apply_rng(hk.transform(sgs_fn))
  optimizer = optax.adam(1e-3)
  params = correction_nn.init(random.PRNGKey(0), np.zeros((1, N2)))
  opt_state = optimizer.init(params)

  @jax.jit
  def loss_fn(params: hk.Params, input: jnp.ndarray, output: jnp.ndarray) -> float:
    predict = correction_nn.apply(params, input)
    return jnp.mean((output - predict)**2)

  @jax.jit
  def evaluate(params: hk.Params, features: jnp.ndarray, labels: jnp.ndarray):
    """Checks the accuracy of predictions compared to labels."""
    logits = correction_nn.apply(params, features)
    predictions = jnp.around(logits, 0)
    return jnp.mean(predictions == labels)
  
  @jax.jit
  def update(params: hk.Params, input: jnp.ndarray, output: jnp.ndarray,
    opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(
      partial(loss_fn, input=input, output=output))(params)
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
      input = train_ds["input"][i: i + batch_size]
      output = train_ds["output"][i: i + batch_size]
      loss, params, opt_state = update(params, input, output, opt_state)
      loss_hist.append(loss)
      desc_str = f"{loss=:.4e}"
      iters.set_description_str(desc_str)

  # a posteriori error estimate
  ks_coarse.run_simulation(ks_fine.x_hist[0, ::r], ks_coarse.CN_FEM)
  im_array = jnp.zeros((3, 1, ks_coarse.x_hist.shape[0], ks_coarse.x_hist.shape[1]))
  im_array = im_array.at[0, 0].set(ks_fine.x_hist[::r])
  im_array = im_array.at[1, 0].set(ks_coarse.x_hist)
  im_array = im_array.at[2, 0].set(ks_coarse.x_hist - ks_fine.x_hist[::r])
  title_array = [f"{N1}", f"{N2}", "diff"]
  plot_with_horizontal_colorbar(
    im_array, fig_size=(4, 6), title_array=title_array,
    file_path=f"results/fig/ks_nu{nu1}_N1{N1}N2{N2}_cmp.pdf"
  )

  corrector = partial(correction_nn.apply, params)
  ks_coarse.run_simulation_with_correction(ks_fine.x_hist[0, ::r], ks_coarse.CN_FEM, corrector)
  im_array = jnp.zeros((3, 1, ks_coarse.x_hist.shape[0], ks_coarse.x_hist.shape[1]))
  im_array = im_array.at[0, 0].set(ks_fine.x_hist[::r])
  im_array = im_array.at[1, 0].set(ks_coarse.x_hist)
  im_array = im_array.at[2, 0].set(ks_coarse.x_hist - ks_fine.x_hist[::r])
  plot_with_horizontal_colorbar(
    im_array, fig_size=(4, 6), title_array=title_array,
    file_path=f"results/fig/ks_nu{nu1}_N1{N1}N2{N2}_correct_cmp.pdf"
  )

if __name__ == "__main__":
  main()