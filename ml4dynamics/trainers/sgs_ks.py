import os
from functools import partial
from typing import Tuple

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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ml4dynamics.dynamics import KS
from ml4dynamics.types import OptState, PRNGKey
from ml4dynamics import utils


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
  if BC == "periodic":
    N1 = config.ks.nx
  elif BC == "Dirichlet-Neumann":
    N1 = config.ks.nx - 1
  r = config.ks.r
  N2 = N1 // r
  dt = config.ks.dt
  rng = random.PRNGKey(config.sim.seed)
  case_num = config.sim.case_num
  sgs_model = config.train.sgs

  # fine simulation
  ks_fine = KS(
    L=L,
    N=N1,
    T=T,
    dt=dt,
    nu=nu,
    c=c,
    BC=BC,
    init_scale=init_scale,
    rng=rng,
  )
  # coarse simulator
  ks_coarse = KS(
    L=L,
    N=N2,
    T=T,
    dt=dt,
    nu=nu,
    c=c,
    BC=BC,
    init_scale=init_scale,
    rng=rng,
  )

  # define the restriction and interpolation operator
  # TODO: try to change the restriction and projection operator to test the
  # results, these operator should have test file
  res_op = jnp.zeros((N2, N1))
  int_op = jnp.zeros((N1, N2))
  res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r + 1].set(1)
  int_op = int_op.at[jnp.arange(N2) * r + 1, jnp.arange(N2)].set(1)

  assert jnp.allclose(res_op @ int_op, jnp.eye(N2))

  # prepare the training data
  if os.path.isfile(
    'data/ks/c{:.1f}T{}n{}_{}.npz'.format(c, T, case_num, sgs_model)
  ):
    data = np.load(
      'data/ks/c{:.1f}T{}n{}_{}.npz'.format(c, T, case_num, sgs_model)
    )
    inputs = data["input"]
    outputs = data["output"]
  else:
    inputs = jnp.zeros((case_num, int(T / dt), N2))
    outputs = jnp.zeros((case_num, int(T / dt), N2))
    for i in range(case_num):
      print(i)
      rng, key = random.split(rng)
      # NOTE: the initialization here is important, DO NOT use the random
      # i.i.d. Gaussian noise as the initial condition
      if BC == "periodic":
        dx = L / N1
        u0 = ks_fine.attractor + init_scale * random.normal(key) *\
          jnp.sin(10 * jnp.pi * jnp.linspace(0, L - L/N1, N1) / L)
      elif BC == "Dirichlet-Neumann":
        dx = L / (N1 + 1)
        x = jnp.linspace(dx, L - dx, N1)
        # different choices of initial conditions
        # u0 = ks_fine.attractor + init_scale * random.normal(key) *\
        #   jnp.sin(10 * jnp.pi * x / L)
        # u0 = random.uniform(key) * jnp.sin(8 * jnp.pi * x / 128) +\
        #   random.uniform(rng) * jnp.sin(16 * jnp.pi * x / 128)
        r0 = random.uniform(key) * 20 + 44
        u0 = jnp.exp(-(x - r0)**2 / r0**2 * 4)
      ks_fine.run_simulation(u0, ks_fine.CN_FEM)
      input = ks_fine.x_hist @ res_op.T  # shape = [step_num, N2]
      output = jnp.zeros_like(input)
      for j in range(ks_fine.step_num):
        if sgs_model == "filter":
          output = ks_fine.x_hist**2 @ res_op.T - input**2
        elif sgs_model == "correction":
          next_step_fine = ks_fine.CN_FEM(
            ks_fine.x_hist[j]
          )  # shape = [N1, step_num]
          next_step_coarse = ks_coarse.CN_FEM(
            input[j]
          )  # shape = [step_num, N2]
          output = output.at[j].set(res_op @ next_step_fine - next_step_coarse)
      inputs = inputs.at[i].set(input)
      outputs = outputs.at[i].set(output)

    inputs = inputs.reshape(-1, N2)
    outputs = outputs.reshape(-1, N2) / dt
    if jnp.any(jnp.isnan(inputs)) or jnp.any(jnp.isnan(outputs)) or\
      jnp.any(jnp.isinf(inputs)) or jnp.any(jnp.isinf(outputs)):
      raise Exception("The data contains Inf or NaN")
    np.savez(
      'data/ks/c{:.1f}T{}n{}_{}.npz'.format(c, T, case_num, sgs_model),
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
    if BC == "periodic":
      dx = L / N2
    elif BC == "Dirichlet-Neumann":
      dx = L / (N2 + 1)
    u = jnp.array(inputs)
    u_x = (jnp.roll(inputs, 1, axis=1) - jnp.roll(inputs, -1, axis=1)) / dx / 2
    u_xx = (
      (jnp.roll(inputs, 1, axis=1) + jnp.roll(inputs, -1, axis=1)) - 2 * inputs
    ) / dx**2
    u_xxxx = ((jnp.roll(inputs, 2, axis=1) + jnp.roll(inputs, -2, axis=1)) -\
        4*(jnp.roll(inputs, 1, axis=1) + jnp.roll(inputs, -1, axis=1)) +\
        6 * inputs) / dx**4
    outputs = outputs.reshape(-1, output_dim)
    if config.train.input == "ux":
      inputs = u_x.reshape(-1, input_dim)
    elif config.train.input == "uxx":
      inputs = u_xx.reshape(-1, input_dim)
    elif config.train.input == "uxxxx":
      inputs = u_xxxx.reshape(-1, input_dim)
    else:
      inputs = u.reshape(-1, input_dim)

  # stratify the data to balance it
  if config.train.stratify == "input_output":
    bins = config.train.bins
    subsample = config.train.subsample
    hist, xedges, yedges = np.histogram2d(
      inputs.reshape(-1), outputs.reshape(-1), bins=bins
    )
    bin_data = {}
    for i in range(bins):
      for j in range(bins):
        if hist[i, j] > 0:
          bin_mask = (xedges[i] <= inputs) & (inputs < xedges[i + 1]) &\
            (yedges[j] <= outputs) & (outputs < yedges[j + 1])
          bin_pts = np.column_stack((inputs[bin_mask], outputs[bin_mask]))
          bin_data[(i, j)] = bin_pts
    stratify_inputs = jnp.zeros((1, 1))
    stratify_outputs = jnp.zeros((1, 1))
    for _ in bin_data.keys():
      if bin_data[_].shape[0] < subsample:
        stratify_inputs = jnp.vstack([stratify_inputs, bin_data[_][:, 0:1]])
        stratify_outputs = jnp.vstack([stratify_outputs, bin_data[_][:, 1:]])
      else:
        samples = np.random.choice(
          jnp.arange(bin_data[_].shape[0]), size=subsample, replace=False
        )
        stratify_inputs = jnp.vstack(
          [stratify_inputs, bin_data[_][samples, 0:1]]
        )
        stratify_outputs = jnp.vstack(
          [stratify_outputs, bin_data[_][samples, 1:]]
        )
    inputs = stratify_inputs
    outputs = stratify_outputs

  if config.train.stratify == "input":
    bins = config.train.bins
    subsample = config.train.subsample
    hist, xedges, yedges = np.histogram2d(
      inputs.reshape(-1), outputs.reshape(-1), bins=bins
    )
    bin_data = {}
    for i in range(bins):
      for j in range(bins):
        if hist[i, j] > 0:
          bin_mask = (xedges[i] <= inputs) & (inputs < xedges[i + 1]) &\
            (yedges[j] <= outputs) & (outputs < yedges[j + 1])
          bin_pts = np.column_stack((inputs[bin_mask], outputs[bin_mask]))
          bin_data[(i, j)] = bin_pts
    stratify_inputs = jnp.zeros((1, 1))
    stratify_outputs = jnp.zeros((1, 1))
    for _ in bin_data.keys():
      if bin_data[_].shape[0] < subsample:
        stratify_inputs = jnp.vstack([stratify_inputs, bin_data[_][:, 0:1]])
        stratify_outputs = jnp.vstack([stratify_outputs, bin_data[_][:, 1:]])
      else:
        samples = np.random.choice(
          jnp.arange(bin_data[_].shape[0]), size=subsample, replace=False
        )
        stratify_inputs = jnp.vstack(
          [stratify_inputs, bin_data[_][samples, 0:1]]
        )
        stratify_outputs = jnp.vstack(
          [stratify_outputs, bin_data[_][samples, 1:]]
        )
    inputs = stratify_inputs
    outputs = stratify_outputs

  # NOTE: visualize and compare the full and stratified dataset
  # bins = config.train.bins
  # subsample = config.train.subsample
  # hist, xedges, yedges = np.histogram2d(
  #   inputs.reshape(-1), outputs.reshape(-1), bins=bins
  # )
  # breakpoint()
  # for _ in range(50):
  #   index = jnp.argmax(hist)
  #   hist[index // bins, index % bins] = 0
  # plt.figure(figsize=(4, 8))
  # plt.subplot(211)
  # plt.scatter(inputs.reshape(-1, 1), outputs, s=.2, c="y")
  # if config.train.stratify:
  #   plt.title(f"{bins} bins; {subsample} subsample; {inputs.shape[0]} samples")
  # plt.subplot(212)
  # # plt.hist2d(inputs.reshape(-1), outputs.reshape(-1), bins=50, density=True)
  # plt.imshow(hist[:, ::-1].T)
  # plt.colorbar()
  # plt.savefig("data.png")
  # breakpoint()

  train_x, test_x, train_y, test_y = train_test_split(
    inputs,
    outputs,
    test_size=0.2,
    random_state=42  #, stratify=outputs
  )
  train_ds = {"input": jnp.array(train_x), "output": jnp.array(train_y)}
  test_ds = {"input": jnp.array(test_x), "output": jnp.array(test_y)}

  # define the network model
  train_mode = config.train.mode
  if train_mode == "regression":
    # training a fully connected neural network to do the closure modeling
    # via regression
    print(f"Fit the SGS model with regression...")

    def sgs_fn(features: jnp.ndarray) -> jnp.ndarray:
      """
      NOTE: an example to show the inconsistency of a priori and a posteriori
      error. Fix the network architecture to be the same, compare the results
      where the lr is chosen to be 1e-3 and 1e-4.
      """

      mlp = hk.Sequential(
        [
          hk.Flatten(),
          hk.Linear(64),
          jax.nn.relu,
          hk.Linear(64),
          jax.nn.relu,
          hk.Linear(output_dim),
        ]
      )
      linear_residue = hk.Linear(output_dim)
      return mlp(features) + linear_residue(features)

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

  elif train_mode == "gaussian":
    # training a fully connected neural network to do the closure modeling
    # by gaussian process
    if config.train.input == "uglobal":
      raise Exception("Gaussian process only supports local modeling!")

    n_g = config.train.n_g
    print(f"Fit the SGS model with {n_g} Gaussian modes...")

    def sgs_fn(features: jnp.ndarray) -> jnp.ndarray:

      mlp = hk.Sequential(
        [
          hk.Flatten(),
          hk.Linear(64),
          jax.nn.relu,
          hk.Linear(64),
          jax.nn.relu,
          hk.Linear(output_dim * n_g * 3),
        ]
      )
      # linear_residue = hk.Linear(output_dim * 2)
      return mlp(features)  # + linear_residue(features)

    def loss_fn(
      params: hk.Params,
      input: jnp.ndarray,
      output: jnp.ndarray,
      rng: PRNGKey,
    ) -> float:
      predict = correction_nn.apply(params, input)
      if n_g == 1:
        # gaussian p.d.f.
        return jnp.mean(
          (output - predict[..., 1:2])**2 / predict[..., 2:]**2 / 2 +\
          jnp.log(jnp.abs(predict[..., 2:]))
        )
      else:
        # gaussian mixture model p.d.f.
        c = jax.nn.softmax(predict[..., :n_g])  # coeff of the GMM
        mean = predict[..., n_g:2 * n_g]
        std = jnp.abs(predict[..., 2 * n_g:]) + 0.01
        # return -jnp.mean(
        #   jax.scipy.special.logsumexp(
        #     a=-((output - mean) / std)**2 / 2,
        #     axis=1,
        #     b=c / std,
        #   )
        # )
        return -jnp.mean(
          jnp.log(
            jnp.sum(c / std * jnp.exp(-((output - mean) / std)**2 / 2), axis=1)
          )
        )

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
    from ml4dynamics.model_jax import model

    print("initialize vae model")
    vae = model(config.train.vae.latents, N2)
    rng, key = random.split(rng)
    params = vae.init(
      key, train_ds["input"][0:1], train_ds["output"][0:1], rng
    )['params']

    lr = 1e-4
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

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

  else:
    raise Exception("Unknown learning mode.")

  lr = config.train.lr
  correction_nn = hk.without_apply_rng(hk.transform(sgs_fn))
  schedule = optax.piecewise_constant_schedule(
    init_value=lr,
    boundaries_and_scales={
      5000: 0.1,
      10000: 0.1,
      #3000: 0.1,
      # 5000: 0.1,
      # 50000: 0.5,
    }
  )
  optimizer = optax.adam(schedule)
  params = correction_nn.init(random.PRNGKey(0), np.zeros((1, input_dim)))
  if config.train.mode == "gaussian":
    params['linear']['b'] = jnp.ones((64, ))
  opt_state = optimizer.init(params)

  # training loop
  loss_hist = []
  epochs = config.train.epochs
  batch_size = config.train.batch_size
  iters = tqdm(range(epochs))
  step = 0
  for _ in iters:
    for i in range(0, len(train_ds["input"]), batch_size):
      rng, key = random.split(rng)
      input = train_ds["input"][i:i + batch_size]
      output = train_ds["output"][i:i + batch_size]
      loss, params, opt_state = update(params, input, output, rng, opt_state)
      lr = schedule(step)
      if train_mode == "regression":
        relative_loss = loss / jnp.mean(output**2)
        desc_str = f"{lr=:.1e}|{relative_loss=:.4e}"
        loss_hist.append(relative_loss)
      elif train_mode == "generative" or train_mode == "gaussian":
        desc_str = f"{lr=:.1e}|{loss=:.4e}"
        loss_hist.append(loss)
      iters.set_description_str(desc_str)
      step += 1
  loss_hist = jnp.array(loss_hist)
  loss_hist = jnp.where(loss_hist > 5, 5, loss_hist)
  plt.plot(jnp.array(loss_hist) - min(loss_hist) + 0.01, label="Loss")
  plt.xlabel("iter")
  plt.yscale("log")
  plt.savefig(f"results/fig/ks_c{c}T{T}n{case_num}_{train_mode}_loss.pdf")
  plt.clf()
  breakpoint()

  valid_loss = 0
  for i in range(0, len(test_ds["input"]), batch_size):
    rng, key = random.split(rng)
    input = train_ds["input"][i:i + batch_size]
    output = train_ds["output"][i:i + batch_size]
    valid_loss += loss_fn(params, input=input, output=output, rng=rng)
  print("lr: {:.2e}".format(lr))
  print("validation loss: {:.4e}".format(valid_loss))

  # A priori analysis: visualize the error distribution
  if config.train.input == "uglobal":
    err = jnp.linalg.norm(correction_nn.apply(params, inputs) - outputs, axis=1)
    err = err.reshape(ks_fine.step_num, -1)
    err = jnp.mean(err, axis=1)
    plt.plot(np.arange(ks_fine.step_num) * ks_fine.dt, err)
    plt.xlabel(r"$T$")
    plt.ylabel("error")
  elif train_mode == "regression" or train_mode == "gaussian":
    if case_num == 1:
      from ml4dynamics.visualize import plot_error_cloudmap
      err = correction_nn.apply(params, inputs) - outputs
      err = err[:, 0]
      plot_error_cloudmap(
        err.reshape(ks_fine.step_num, N2).T,
        u.reshape(ks_fine.step_num, N2).T, u_x.T, u_xx.T, u_xxxx.T, train_mode
      )

    # TODO: the treatment here is temporary
    step = jnp.prod(jnp.array(inputs.shape)) // N2 // case_num
    t = jnp.linspace(0, T, step).reshape(1, -1, 1)
    t = jnp.tile(t, (case_num, 1, N2)).reshape(-1)
    plt.figure(figsize=(16, 4))
    plt.subplot(141)
    if config.train.stratify:
      plt.scatter(inputs.reshape(-1, 1), outputs, s=.2, c="y")
    else:
      plt.scatter(inputs.reshape(-1, 1), outputs, s=.2, c=t)
    # x_min = inputs.min()
    # x_max = inputs.max()
    # y_min = outputs.min()
    # y_max = outputs.max()
    # for x in xedges:
    #   plt.plot(
    #     x * jnp.ones(100), jnp.linspace(y_min, y_max, 100), linewidth=.5, c="b"
    #   )
    # for y in yedges:
    #   plt.plot(
    #     jnp.linspace(x_min, x_max, 100), y * jnp.ones(100), linewidth=.5, c="b"
    #   )
    tmp_input = jnp.linspace(jnp.min(inputs), jnp.max(inputs), 20)
    tmp_output = correction_nn.apply(params, tmp_input.reshape(-1, 1))
    if train_mode == "regression":
      plt.plot(tmp_input, tmp_output, label="learned", c="r")
    elif train_mode == "gaussian":
      for _ in range(n_g):
        plt.errorbar(
          tmp_input,
          tmp_output[:, n_g + _],
          yerr=jnp.abs(tmp_output[:, 2 * n_g + _]),
          fmt='o-',
          markersize=.5,
          label="learned"
        )
    plt.title("loss = {:.4e}".format(loss_hist[-1]))
    plt.subplot(142)
    plt.hist2d(inputs.reshape(-1), outputs.reshape(-1), bins=20, density=True)
    plt.title(f"{bins} bins; {subsample} subsample; {inputs.shape[0]} samples")
    plt.colorbar()
    plt.subplot(143)
    plt.hist(inputs, bins=200, label=config.train.input, density=True)
    # plt.yscale("log")
    plt.legend()
    plt.subplot(144)
    plt.hist(outputs, bins=200, label=r"$\tau$", density=True)
    # plt.yscale("log")
  plt.legend()
  plt.savefig(
    f"results/fig/ks_c{c}T{T}n{case_num}_{train_mode}_{config.train.input}_scatter.png",
    dpi=1000
  )
  plt.clf()

  utils.a_posteriori_analysis(config, )
  breakpoint()


if __name__ == "__main__":

  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  main(config_dict)
