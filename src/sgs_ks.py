from typing import Iterator, Optional, Tuple

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
from tqdm import tqdm

from src.dynamics import KS
from src.types import Batch, OptState, PRNGKey

def main():

  # model parameters
  nu1 = .5
  c = .1
  L = 10 * jnp.pi
  T = 40
  # solver parameters
  N1 = 2048
  N2 = 1024
  dt = 0.01
  r = N1 // N2

  # fine simulation
  ks_fine = KS(  
    N = N1, T = T, dt = dt, dx = L / (N1+1), tol = 1e-8, init_scale = 1e-2,
    tv_scale = 1e-8, L = L, nu = nu1, c = c
  )
  ks_fine.run_simulation(ks_fine.x_targethist[:,0], ks_fine.CN_FEM)

  # coarse simulation
  ks_coarse = KS(
    N = N2, T = T, dt = dt, dx = L / (N2 + 1), tol = 1e-8, init_scale = 1e-2,
    tv_scale = 1e-8, L = L, nu = nu1, c = c
  )
  ks_coarse.run_simulation(ks_fine.x_hist[::r, 0], ks_coarse.CN_FEM)

  if False:
    # visualize the difference between the fine and coarse grid simulation
    plt.subplot(3, 1, 1)
    plt.imshow(ks_fine.x_hist[::r])
    plt.colorbar()
    plt.title(f"n = {N1}")
    plt.axis("off")
    plt.subplot(3, 1, 2)
    plt.imshow(ks_coarse.x_hist)
    plt.colorbar()
    plt.title(f"n = {N2}")
    plt.axis("off")
    plt.subplot(3, 1, 3)
    plt.imshow(ks_coarse.x_hist - ks_fine.x_hist[::r])
    plt.colorbar()
    plt.title("diff")
    plt.axis("off")
    plt.savefig(f"results/fig/ks_nu{nu1}_N1{N1}N2{N2}_cmp.pdf")

  # define the restriction and interpolation operator
  res_op = jnp.zeros((N2, N1))
  int_op = jnp.zeros((N1, N2))
  res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r].set(1)
  int_op = int_op.at[jnp.arange(N2) * r, jnp.arange(N2)].set(1)

  assert jnp.allclose(res_op @ int_op, jnp.eye(N2))

  input = (res_op @ ks_fine.x_hist).T # shape = [step_num, N2]
  output = jnp.zeros_like(input)
  for i in range(ks_fine.step_num):
    next_step_fine = ks_fine.CN_FEM(ks_fine.x_hist[:,i]) # shape = [step_num, N1]
    next_step_coarse = ks_coarse.CN_FEM(input[i]) # shape = [step_num, N2]
    output = output.at[i].set(res_op @ next_step_fine - next_step_coarse)

  np.savez('data/ks/tmp.npz', input = input, output = output)


  # training a fully connected neural network to do the closure modeling
  def sgs_fn(features: jnp.ndarray) -> jnp.ndarray:
    """Standard LeNet-300-100 MLP network."""

    mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(2048), jax.nn.relu,
      hk.Linear(1024), jax.nn.relu,
      hk.Linear(N2), jax.nn.sigmoid
    ])
    return mlp(features)
  
  correction_nn = hk.without_apply_rng(hk.transform(sgs_fn))
  optimizer = optax.adam(1e-3)
  params = correction_nn.init(key, np.zeros((1, N2)))
  opt_state = optimizer.init(params)

  @jax.jit
  def loss_fn(params: hk.Params, input: jnp.ndarray, output: jnp.ndarray) -> float:
    predict = correction_nn.apply(params, input)
    return optax.l2_loss(output, predict)

  @jax.jit
  def evaluate(params: hk.Params, features: jnp.ndarray, labels: jnp.ndarray):
    """Checks the accuracy of predictions compared to labels."""
    logits = correction_nn.apply(params, features)
    predictions = jnp.around(logits, 0)
    return jnp.mean(predictions == labels)
  
  @jax.jit
  def update(params: hk.Params, rng: PRNGKey,
             opt_state: OptState) -> Tuple[Array, hk.Params, OptState]:
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_fn)(params, input, output)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state
  
  loss_hist = []
  epochs = 10000
  iters = tqdm(range(epochs))
  

  for step in iters:

    loss, params, opt_state = update(params, key, lambda_, opt_state)
    loss_hist.append(loss)

    if step % FLAGS.eval_frequency == 0:
      desc_str = f"{loss=:.4e}"

      key, rng = jax.random.split(rng)
      # wasserstein distance
      if FLAGS.case == "wasserstein":
        KL = density_fit_rkl_loss_fn(params, rng, lambda_, FLAGS.batch_size)
        kin = loss - KL * lambda_
        desc_str += f"{KL=:.4f} | {kin=:.1f} | {lambda_=:.1f}"
      elif FLAGS.case == "mfg":
        # KL = reverse_kl_loss_fn(params, rng, 0, FLAGS.batch_size)
        KL = kl_loss_fn(params, rng, 0, FLAGS.batch_size)
        pot = potential_loss_fn(params, rng, T, FLAGS.batch_size)
        kin = loss - KL * lambda_ - pot
        desc_str += f"{KL=:.4f} | {pot=:.2f} | {kin=:.2f} | {lambda_=:.1f}"

      iters.set_description_str(desc_str)

if __name__ == "__main__":
  main()