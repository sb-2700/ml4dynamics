import os
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
  nu = config.ks.nu
  c = config.ks.c
  L = config.ks.L
  T = config.ks.T
  init_scale = config.ks.init_scale
  BC = config.ks.BC
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
    init_scale=init_scale,
    L=L,
    nu=nu,
    c=c,
    BC=BC,
    key=key,
  )
  # coarse simulator
  ks_coarse = KS(
    N=N2,
    T=T,
    dt=dt,
    init_scale=init_scale,
    L=L,
    nu=nu,
    c=c,
    BC=BC,
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

  # prepare the training data
  if os.path.isfile(
    'data/ks/nu{:.1f}_c{:.1f}_n{}.npz'.format(nu, c, config.sim.case_num)
  ):
    data = np.load('data/ks/nu{:.1f}_c{:.1f}_n{}.npz'.format(nu, c, config.sim.case_num))
    inputs = data["input"]
    outputs = data["output"]
  else:
    inputs = jnp.zeros((config.sim.case_num, int(T / dt), N2))
    outputs = jnp.zeros((config.sim.case_num, int(T / dt), N2))
    for i in range(config.sim.case_num):
      print(i)
      key, subkey = random.split(key)
      # NOTE: the initialization here is important, DO NOT use the random i.i.d.
      # Gaussian noise as the initial condition
      x = ks_fine.attractor + init_scale * random.normal(subkey) *\
        jnp.sin(5 * jnp.linspace(0, L - L/N1, N1))
      ks_fine.run_simulation(x, ks_fine.CN_FEM)
      input = ks_fine.x_hist @ res_op.T  # shape = [step_num, N2]
      output = jnp.zeros_like(input)
      for j in range(ks_fine.step_num):
        next_step_fine = ks_fine.CN_FEM(
          ks_fine.x_hist[j]
        )  # shape = [N1, step_num]
        next_step_coarse = ks_coarse.CN_FEM(input[j])  # shape = [step_num, N2]
        output = output.at[j].set(res_op @ next_step_fine - next_step_coarse)
      inputs = inputs.at[i].set(input)
      outputs = outputs.at[i].set(output)

    inputs = inputs.reshape(-1, N2)
    outputs = outputs.reshape(-1, N2) / dt
    if jnp.any(jnp.isnan(inputs)) or jnp.any(jnp.isnan(outputs)) or\
      jnp.any(jnp.isinf(inputs)) or jnp.any(jnp.isinf(outputs)):
      raise Exception("The data contains Inf or NaN")
    np.savez(
      'data/ks/nu{:.1f}_c{:.1f}_n{}.npz'.format(nu, c, config.sim.case_num),
      input=inputs,
      output=outputs
    )

  # preprocess the input-output pair for training
  if config.train.input == "uglobal":
    # training global ROM
    input_dim = output_dim = N2
    inputs = inputs.reshape(-1, input_dim)
    outputs = outputs.reshape(-1, output_dim)
  else:
    # training local ROM with local input
    input_dim = output_dim = 1
    dx = L / N2
    u_x = (jnp.roll(inputs, 1, axis=1) - jnp.roll(inputs, -1, axis=1)) / dx / 2
    u_xx = (
      (jnp.roll(inputs, 1, axis=1) + jnp.roll(inputs, -1, axis=1)) - 2 * inputs
    ) / dx**2
    u_xxxx = ((jnp.roll(inputs, 2, axis=1) + jnp.roll(inputs, -2, axis=1)) -\
        4*(jnp.roll(inputs, 1, axis=1) + jnp.roll(inputs, -1, axis=1)) + 6 * inputs) /\
        dx**4
    outputs = outputs.reshape(-1, output_dim)
    if config.train.input == "ux":
      inputs = u_x.reshape(-1, input_dim)
    elif config.train.input == "uxx":
      inputs = u_xx.reshape(-1, input_dim)
    elif config.train.input == "uxxxx":
      inputs = u_xxxx.reshape(-1, input_dim)
    else: 
      inputs = inputs.reshape(-1, input_dim)

  # train test split
  train_x, test_x, train_y, test_y = train_test_split(
    inputs, outputs, test_size=0.2, random_state=42
  )
  train_ds = {"input": jnp.array(train_x), "output": jnp.array(train_y)}
  test_ds = {"input": jnp.array(test_x), "output": jnp.array(test_y)}

  # define the network model
  train_mode = config.train.mode
  if train_mode == "regression":
    # training a fully connected neural network to do the closure modeling
    # via regression
    def sgs_fn(features: jnp.ndarray) -> jnp.ndarray:
      """
      NOTE: an example to show the inconsistency of a priori and a posteriori
      error. Fix the network architecture to be the same, compare the results
      where the lr is chosen to be 1e-3 and 1e-4.
      """

      mlp = hk.Sequential(
        [
          hk.Flatten(),
          hk.Linear(512),
          jax.nn.relu,
          hk.Linear(512),
          jax.nn.relu,
          hk.Linear(output_dim),
        ]
      )
      # linear_residue = hk.Linear(output_dim)
      return mlp(features) # + linear_residue(features)

    correction_nn = hk.without_apply_rng(hk.transform(sgs_fn))
    # 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-5
    lr = 5e-4
    optimizer = optax.adam(lr)
    params = correction_nn.init(random.PRNGKey(0), np.zeros((1, input_dim)))
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(
      params: hk.Params,
      input: jnp.ndarray,
      output: jnp.ndarray,
      rng: PRNGKey,
    ) -> float:
      predict = correction_nn.apply(params, input)
      return jnp.mean((output - predict)**2)

    @jax.jit
    def update(
      params: hk.Params, input: jnp.ndarray, output: jnp.ndarray, rng: PRNGKey,
      opt_state: OptState
    ) -> Tuple[Array, hk.Params, OptState]:
      """Single SGD update step."""
      loss, grads = jax.value_and_grad(
        partial(loss_fn, input=input, output=output, rng=rng)
      )(params)
      # loss, grads = jax.value_and_grad(loss_fn)(params, input, output)
      updates, new_opt_state = optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return loss, new_params, new_opt_state

  elif train_mode == "generative":
    """
    TODO: currently we are using different structure for regression and
    generative modeling, need to unify
    """
    from src.model_jax import model

    print("initialize vae model")
    vae = model(config.train.vae.latents, N2)
    rng, key = random.split(key)
    params = vae.init(
      key, train_ds["input"][0:1], train_ds["output"][0:1], rng
    )['params']

    lr = 1e-4
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(
      params: hk.Params, input: jnp.ndarray, output: jnp.ndarray, rng: PRNGKey
    ) -> float:

      # NOTE: notice the input is the condition, output is x
      recon_x, mean, logvar = vae.apply({"params": params}, output, input, rng)
      # NOTE: lots of code are using BCE, we may also try
      return jnp.sum((input - recon_x)**2) +\
        -0.5 * jnp.sum(1 + logvar - jnp.power(mean, 2) - jnp.exp(logvar))

    @jax.jit
    def update(
      params: hk.Params, input: jnp.ndarray, output: jnp.ndarray, rng: PRNGKey,
      opt_state: OptState
    ) -> Tuple[Array, hk.Params, OptState]:
      """Single SGD update step."""
      loss, grads = jax.value_and_grad(
        partial(loss_fn, input=input, output=output, rng=rng)
      )(params)
      # loss, grads = jax.value_and_grad(loss_fn)(params, input, output)
      updates, new_opt_state = optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return loss, new_params, new_opt_state
  
  elif train_mode == "gaussian":
    # training a fully connected neural network to do the closure modeling
    # by gaussian process regression
    if config.train.input == "uglobal":
      raise Exception("GPR only supports local modeling!")

    def sgs_fn(features: jnp.ndarray) -> jnp.ndarray:
      """
      NOTE: an example to show the inconsistency of a priori and a posteriori
      error. Fix the network architecture to be the same, compare the results
      where the lr is chosen to be 1e-3 and 1e-4.
      """

      mlp = hk.Sequential(
        [
          hk.Flatten(),
          hk.Linear(512),
          jax.nn.relu,
          hk.Linear(512),
          jax.nn.relu,
          hk.Linear(output_dim * 2),
        ]
      )
      # linear_residue = hk.Linear(output_dim * 2)
      return mlp(features) # + linear_residue(features)

    correction_nn = hk.without_apply_rng(hk.transform(sgs_fn))
    # 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-5
    lr = 5e-4
    optimizer = optax.adam(lr)
    params = correction_nn.init(random.PRNGKey(0), np.zeros((1, input_dim)))
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(
      params: hk.Params,
      input: jnp.ndarray,
      output: jnp.ndarray,
      rng: PRNGKey,
    ) -> float:
      predict = correction_nn.apply(params, input)
      return jnp.mean((output - predict[0])**2 / predict[1]**2 / 2 +
                       jnp.log(predict[1]))

    @jax.jit
    def update(
      params: hk.Params, input: jnp.ndarray, output: jnp.ndarray, rng: PRNGKey,
      opt_state: OptState
    ) -> Tuple[Array, hk.Params, OptState]:
      """Single SGD update step."""
      loss, grads = jax.value_and_grad(
        partial(loss_fn, input=input, output=output, rng=rng)
      )(params)
      # loss, grads = jax.value_and_grad(loss_fn)(params, input, output)
      updates, new_opt_state = optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return loss, new_params, new_opt_state

  else:
    raise Exception("Unknown learning mode.")

  # training loop
  loss_hist = []
  epochs = config.train.epochs
  batch_size = config.train.batch_size
  iters = tqdm(range(epochs))
  for _ in iters:
    for i in range(0, len(train_ds["input"]), batch_size):
      rng, key = random.split(key)
      input = train_ds["input"][i:i + batch_size]
      output = train_ds["output"][i:i + batch_size]
      loss, params, opt_state = update(params, input, output, rng, opt_state)
      loss_hist.append(loss)
      if train_mode == "regression":
        relative_loss = loss / jnp.mean(output**2)
        desc_str = f"{relative_loss=:.4e}"
      elif train_mode == "generative" or train_mode == "gaussian":
        desc_str = f"{loss=:.4e}"
      iters.set_description_str(desc_str)

  valid_loss = 0
  for i in range(0, len(test_ds["input"]), batch_size):
    rng, key = random.split(key)
    input = train_ds["input"][i:i + batch_size]
    output = train_ds["output"][i:i + batch_size]
    valid_loss += loss_fn(params, input=input, output=output, rng=rng)
  print("lr: {:.2e}".format(lr))
  print("validation loss: {:.4e}".format(valid_loss))

  # A priori analysis
  # visualize the error distribution
  if config.train.input == "uglobal":
    err = jnp.linalg.norm(correction_nn.apply(params, inputs) - outputs, axis=1)
    err = err.reshape(1000, -1)
    err = jnp.mean(err, axis=1)
    plt.plot(np.arange(ks_fine.step_num) * ks_fine.dt, err)
    plt.xlabel(r"$T$")
    plt.ylabel("error")
  elif config.train.mode == "regression":
    err = correction_nn.apply(params, inputs) - outputs
    from src.visualize import plot_error_cloudmap
    plot_error_cloudmap(
      inputs.reshape(1000, 256).T,
      err.reshape(1000, 256).T, u_x.T, u_xx.T, u_xxxx.T
    )
    if config.train.input == "u":
      plt.scatter(inputs.reshape(-1, 1), outputs, s=.2, c=err)
    elif config.train.input == "ux":
      plt.scatter(u_x.reshape(-1, 1), outputs, s=.2, c=err)
    elif config.train.input == "uxx":
      plt.scatter(u_xx.reshape(-1, 1), outputs, s=.2, c=err)
    elif config.train.input == "uxxxx":
      plt.scatter(u_xxxx.reshape(-1, 1), outputs, s=.2, c=err)
  plt.savefig(f"results/fig/{config.train.input}_tau_err_scatter.pdf")
  plt.clf()

  # a posteriori analysis
  key, subkey = random.split(key)
  x = ks_fine.attractor + init_scale * random.normal(subkey) *\
        jnp.sin(5 * jnp.linspace(0, L - L/N1, N1))
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
    file_path=
    f"results/fig/ks_nu{nu}_N{N1}n{config.sim.case_num}_{train_mode}_cmp.pdf"
  )
  print(
    "rmse without correction: {:.4f}".format(
      jnp.sqrt(jnp.mean((ks_coarse.x_hist.T - ks_fine.x_hist[:, ::r].T)**2))
    )
  )
  baseline = ks_coarse.x_hist

  if train_mode == "regression":

    def corrector(input):
      """
      input.shape = (N, )
      output.shape = (N, )
      """

      # dx = L / N2
      # u_x = (jnp.roll(input, 1) - jnp.roll(input, -1)) /dx/2
      # u_xx = ((jnp.roll(input, 1) + jnp.roll(input, -1)) - 2 * input) / dx**2
      # u_xxxx = ((jnp.roll(input, 2) + jnp.roll(input, -2)) -\
      #     4*(jnp.roll(input, 1) + jnp.roll(input, -1)) + 6 * input) /\
      #     dx**4
      # # local model: [1] to [1]
      # return partial(correction_nn.apply, params)(input.reshape(-1,
      #                                                          1)).reshape(-1)
      # global model: [N] to [N]
      return partial(correction_nn.apply, params)(
        input.reshape(1, -1)
      ).reshape(-1)

  elif train_mode == "generative":
    vae_bind = vae.bind({"params": params})
    z = random.normal(key, shape=(1, config.train.vae.latents))
    corrector = partial(vae_bind.generate, z)
  elif train_mode == "gaussian":
    z = random.normal(key)
    def corrector(input):
      """
      input.shape = (N, )
      output.shape = (N, )
      """

      dx = L / N2
      u_x = (jnp.roll(input, 1) - jnp.roll(input, -1)) /dx/2
      u_xx = ((jnp.roll(input, 1) + jnp.roll(input, -1)) - 2 * input) / dx**2
      u_xxxx = ((jnp.roll(input, 2) + jnp.roll(input, -2)) -\
          4*(jnp.roll(input, 1) + jnp.roll(input, -1)) + 6 * input) /\
          dx**4
      # local model: [1] to [1]
      tmp = partial(correction_nn.apply, params)(u_xx.reshape(-1, 1))
      return (tmp[:, 0] + tmp[:, 1] * z).reshape(-1)

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
    file_path=
    f"results/fig/ks_nu{nu}_N{N1}n{config.sim.case_num}_{train_mode}_correct_cmp.pdf"
  )
  print(
    "rmse with correction: {:.4f}".format(
      jnp.sqrt(jnp.mean((ks_coarse.x_hist.T - ks_fine.x_hist[:, ::r].T)**2))
    )
  )

  # compare the simulation statistics (A posteriori analysis)
  from src.visualize import plot_stats
  plot_stats(
    np.arange(ks_fine.x_hist.shape[0]) * ks_fine.dt,
    ks_fine.x_hist,
    baseline,
    ks_coarse.x_hist,
    f"results/fig/ks_nu{nu}_N{N1}n{config.sim.case_num}_{train_mode}_cmp_stats.pdf",
  )


if __name__ == "__main__":

  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
