import gc
from functools import partial
from time import time

import h5py
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
import ml_collections
import numpy as np
import optax
import pickle
import torch
from box import Box
from jax import random as random
from jax.numpy.linalg import solve
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ml4dynamics import dynamics
from ml4dynamics.dataset_utils.dataset_utils import res_int_fn
from ml4dynamics.models.models_jax import CustomTrainState, UNet
from ml4dynamics.trainers import train_utils
from ml4dynamics.types import PRNGKey
from ml4dynamics.visualize import plot_stats_aux

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)


def load_data(
  config_dict: ml_collections.ConfigDict,
  batch_size: int,
  num_workers: int = 16,
  mode: str = "jax"
):

  config = Box(config_dict)
  pde_type = config.case
  case_num = config.sim.case_num
  if pde_type == "react_diff":
    alpha = config.sim.alpha
    beta = config.sim.beta
    gamma = config.sim.gamma
    sgs = config.train.sgs
    dataset = "alpha{:.2f}_beta{:.2f}_gamma{:.2f}_n{}_{}".format(
      alpha, beta, gamma, case_num, sgs
    )
  elif pde_type == "ns_channel":
    Re = config.sim.Re
    nx = config.sim.nx
    BC = config.sim.BC
    dataset = f"{BC}_Re{Re}_nx{nx}_n{case_num}"
  elif pde_type == "ns_hit":
    Re = config.sim.Re
    n = config.sim.n // config.sim.r
    dataset = f"Re{Re}_nx{n}_n{case_num}"
  elif pde_type == "ks":
    c = config.sim.c
    T = config.sim.T
    sgs_model = config.train.sgs
    dataset = f"c{c:.1f}_T{T}_n{case_num}_{sgs_model}"
  h5_filename = f"data/{pde_type}/{dataset}.h5"

  with h5py.File(h5_filename, "r") as h5f:
    inputs = h5f["data"]["inputs"][()]
    outputs = h5f["data"]["outputs"][()]
    if mode == "torch":
      inputs = inputs.transpose(0, 3, 1, 2)
      outputs = outputs.transpose(0, 3, 1, 2)
      GPU = 0
      device = torch.device(
        "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"
      )
      inputs = torch.from_numpy(inputs).to(device)
      outputs = torch.from_numpy(outputs).to(device)
  if pde_type == "ks" and config.sim.BC == "Dirichlet-Neumann":
    padding = np.zeros((inputs.shape[0], 1, 1))
    inputs = np.concatenate([inputs, padding], axis=1)
    outputs = np.concatenate([outputs, padding], axis=1)
  train_x, test_x, train_y, test_y = train_test_split(
    inputs, outputs, test_size=0.2, random_state=config.sim.seed
  )
  train_dataset = TensorDataset(
    torch.tensor(train_x, dtype=torch.float64),
    torch.tensor(train_y, dtype=torch.float64)
  )
  train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True  # , num_workers=num_workers
  )
  test_dataset = TensorDataset(
    torch.tensor(test_x, dtype=torch.float64),
    torch.tensor(test_y, dtype=torch.float64)
  )
  test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True  # , num_workers=num_workers
  )
  return inputs, outputs, train_dataloader, test_dataloader, dataset


def create_fine_coarse_simulator(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  rng = random.PRNGKey(config.sim.seed)
  T = config.sim.T
  dt = config.sim.dt
  r = config.sim.r
  L = config.sim.L
  n = config.sim.n
  # model parameters
  if config.case == "react_diff":
    d = config.sim.d
    model_fine = dynamics.react_diff(
      L=L,
      N=n**2 * 2,
      T=T,
      dt=dt,
      alpha=config.sim.alpha,
      beta=config.sim.beta,
      gamma=config.sim.gamma,
      d=d,
    )
    model_coarse = dynamics.react_diff(
      L=L,
      N=(n // r)**2 * 2,
      T=T,
      dt=dt,
      alpha=config.sim.alpha,
      beta=config.sim.beta,
      gamma=config.sim.gamma,
      d=d,
    )
  elif config.case == "ks":
    c = config.sim.c
    init_scale = config.sim.init_scale
    BC = config.sim.BC
    # solver parameters
    if BC == "periodic":
      N1 = n
    elif BC == "Dirichlet-Neumann":
      N1 = n - 1
    N2 = N1 // r
    # fine simulation
    model_fine = dynamics.KS(
      L=L,
      N=N1,
      T=T,
      dt=dt,
      nu=config.sim.nu,
      c=c,
      BC=BC,
      init_scale=init_scale,
      rng=rng,
    )
    # coarse simulator
    model_coarse = dynamics.KS(
      L=L,
      N=N2,
      T=T,
      dt=dt,
      nu=config.sim.nu,
      c=c,
      BC=BC,
      init_scale=init_scale,
      rng=rng,
    )
  elif config.case == "ns_hit":
    model_fine = dynamics.ns_hit(
      L=L * np.pi,
      N=n,
      T=T,
      dt=dt,
      nu=1 / config.sim.Re,
      init_scale=n**(1.5),
    )
    model_coarse = dynamics.ns_hit(
      L=L * np.pi,
      N=n // r,
      T=T,
      dt=dt,
      nu=1 / config.sim.Re,
      init_scale=(n // r)**(1.5),
    )
  return model_fine, model_coarse


def create_ns_channel_simulator(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  # model parameters
  model = dynamics.ns_channel(
    Lx=config.sim.Lx,
    nx=config.sim.nx,
    ny=config.sim.ny,
    T=config.sim.T,
    dt=config.sim.dt,
    Re=config.sim.Re,
  )
  return model


def create_ns_hit_simulator(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  # model parameters
  model = dynamics.ns_hit(
    L=config.sim.L * np.pi,
    N=config.sim.n,
    T=config.sim.T,
    dt=config.sim.dt,
    nu=1 / config.sim.Re,
    init_scale=config.sim.n**(1.5)
  )
  return model


def prepare_unet_train_state(
  config_dict: ml_collections.ConfigDict, load_dict: str = None,
  is_training: bool = True
):

  config = Box(config_dict)
  rng = random.PRNGKey(config.sim.seed)
  if config.case == "react_diff":
    input_features = 2
    output_features = 2
    nx = ny = config.sim.n
    n_sample = int(config.sim.T / config.sim.dt * 0.8)
    DIM = 2
  elif config.case == "ns_channel":
    input_features = 2
    output_features = 1
    n_sample = int(config.sim.T / config.sim.dt * 0.8)
    nx = config.sim.nx
    ny = config.sim.ny
    DIM = 2
  elif config.case == "ns_hit":
    input_features = 1
    output_features = 1
    n_sample = int(config.sim.T / config.sim.dt * 0.8)
    nx = ny = config.sim.n
    DIM = 2
  elif config.case == "ks":
    input_features = 1
    output_features = 1
    n_sample = int(config.sim.T / config.sim.dt * 0.8)
    nx = config.sim.n
    DIM = 1
  unet = UNet(
    input_features=input_features, output_features=output_features,
    DIM=DIM, training = is_training,
  )
  rng1, rng2 = random.split(rng)
  init_rngs = {'params': rng1, 'dropout': rng2}
  if load_dict:
    with open(f"ckpts/{load_dict}.pkl", "rb") as f:
      unet_variables = pickle.load(f)
  else:
    if config.case == "ks":
      # init 1D UNet
      unet_variables = unet.init(init_rngs, jnp.ones([1, nx, input_features]))
    else:
      unet_variables = unet.init(init_rngs, jnp.ones([1, nx, ny, input_features]))
  step_per_epoch = n_sample * config.sim.case_num // config.train.batch_size_unet
  # TODO: need to specify the scheduler here for different training
  schedule = optax.piecewise_constant_schedule(
    init_value=config.train.lr,
    boundaries_and_scales={
      int(b): 0.1
      for b in jnp.arange(
        config.train.decay * step_per_epoch, config.train.epochs *
        step_per_epoch, config.train.decay * step_per_epoch
      )
    }
  )
  optimizer = optax.adam(schedule)
  train_state = CustomTrainState.create(
    apply_fn=unet.apply,
    params=unet_variables["params"],
    tx=optimizer,
    batch_stats=unet_variables["batch_stats"]
  )
  return train_state, schedule


def data_stratification(config_dict: ml_collections.ConfigDict):
  config = Box(config_dict)
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


def eval_a_priori(
  train_state: CustomTrainState,
  train_dataloader: DataLoader,
  test_dataloader: DataLoader,
  inputs: jnp.ndarray,
  outputs: jnp.ndarray,
  dim: int = 2,
  fig_name: str = None,
):

  total_loss = 0
  count = 0
  for batch_inputs, batch_outputs in train_dataloader:
    predict, _ = train_state.apply_fn_with_bn(
      {
        "params": train_state.params,
        "batch_stats": train_state.batch_stats
      },
      jnp.array(batch_inputs),
      is_training=False
    )
    loss = jnp.mean((predict - jnp.array(batch_outputs))**2)
    total_loss += loss
    count += 1
  print(f"train loss: {total_loss/count:.4e}")
  total_loss = 0
  count = 0
  for batch_inputs, batch_outputs in test_dataloader:
    predict, _ = train_state.apply_fn_with_bn(
      {
        "params": train_state.params,
        "batch_stats": train_state.batch_stats
      },
      jnp.array(batch_inputs),
      is_training=False
    )
    loss = jnp.mean((predict - jnp.array(batch_outputs))**2)
    total_loss += loss
    count += 1
  print(f"test loss: {total_loss/count:.4e}")

  if dim == 1:
    predicts, _ = train_state.apply_fn_with_bn(
      {
        "params": train_state.params,
        "batch_stats": train_state.batch_stats
      },
      jnp.array(inputs),
      is_training=False
    )
    im_array = np.zeros((3, 1, *(outputs[..., 0].T).shape))
    im_array[0, 0] = outputs[..., 0].T
    im_array[1, 0] = predicts[..., 0].T
    im_array[2, 0] = (outputs - predicts)[..., 0].T
    plot_with_horizontal_colorbar(
      im_array, (12, 6), None, f"results/fig/{fig_name}.png", 100
    )
  elif dim == 2:
    # visualization
    n_plot = 4
    index_array = np.arange(
      0, n_plot * outputs.shape[0] // n_plot - 1, outputs.shape[0] // n_plot
    )
    im_array = np.zeros((3, n_plot, *(outputs[0, ..., 0].T).shape))
    for j in range(n_plot):
      predicts, _ = train_state.apply_fn_with_bn(
        {
          "params": train_state.params,
          "batch_stats": train_state.batch_stats
        },
        jnp.array(inputs[index_array[j]:index_array[j] + 1]),
        is_training=False
      )
      im_array[0, j] = outputs[index_array[j], ..., 0].T
      im_array[1, j] = predicts[index_array[j], ..., 0].T
      im_array[2, j] = (outputs - predicts)[index_array[j], ..., 0].T
    plot_with_horizontal_colorbar(
      im_array, (12, 6), None, f"results/fig/{fig_name}.png", 100
    )

    im_array = np.zeros((2, n_plot, *(outputs[0, ..., 0].T).shape))
    title_array1 = []
    title_array2 = []
    for j in range(n_plot):
      im_array[0, j] = inputs[index_array[j], ..., 0].T
      im_array[1, j] = outputs[index_array[j], ..., 0].T
      title_array1.append(f"{inputs[index_array[j], ..., 0].max():.1e}")
      title_array2.append(f"{outputs[index_array[j], ..., 0].max():.1e}")
    plot_with_horizontal_colorbar(
      im_array, (12, 6), title_array1+title_array2, f"results/fig/dataset1.png", 100
    )
    plt.hist(inputs.reshape(-1), bins=100, density=True, label="inputs")
    plt.hist(outputs.reshape(-1), bins=100, density=True, label="outputs")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"results/fig/dataset2.png")
    plt.close()


def eval_a_posteriori(
  config_dict: ml_collections.ConfigDict,
  train_state: CustomTrainState,
  inputs: jnp.ndarray,
  outputs: jnp.ndarray,
  dim: int = 2,
  fig_name: str = None,
):

  config = Box(config_dict)
  beta = 0.0
  if config.case == "ns_channel":
    model = create_ns_channel_simulator(config)
    run_simulation = partial(
      train_utils.run_ns_simulation_pressue_correction, train_state, model,
      outputs, beta
    )
  else:
    _, model = create_fine_coarse_simulator(config)
    if config.case == "react_diff":
      iter_ = model.adi
      type_ = None
    elif config.case == "ns_hit":
      run_simulation = partial(
        train_utils.run_simulation_sgs, train_state, model, model.CN_real,
        outputs, beta, None
      )
    elif config.case == "ks":
      iter_ = model.CN_FEM
      type_ = "pad"
    if config.train.sgs == "fine_correction":
      res_fn, int_fn = res_int_fn(config)
      run_simulation = partial(
        train_utils.run_simulation_fine_grid_correction, train_state, model,
        iter_, outputs, beta, res_fn, int_fn, type_
      )
    elif config.train.sgs == "filter":
      run_simulation = partial(
        train_utils.run_simulation_sgs, train_state, model, iter_, outputs,
        beta, type_
      )
    elif config.train.sgs == "coarse_correction":
      run_simulation = partial(
        train_utils.run_simulation_coarse_grid_correction,
        train_state, model, iter_, outputs, beta, type_
      )

  start = time()
  step_num = model.step_num
  inputs = inputs[:step_num]
  outputs = outputs[:step_num]
  x_hist = run_simulation(inputs[0])
  # NOTE: for general a-posteriori test
  # u_fft = jnp.zeros((2, nx, nx))
  # u_fft = u_fft.at[:, :10, :10].set(
  #   random.normal(key=random.PRNGKey(0), shape=(2, 10, 10))
  # )
  # uv0 = jnp.real(jnp.fft.fftn(u_fft, axes=(1, 2))) / nx
  print(f"simulation takes {time() - start:.2f}s...")
  if jnp.any(jnp.isnan(x_hist)) or jnp.any(jnp.isinf(x_hist)):
    print("similation contains NaN!")
  # print(
  #   "L2 error: x{:.4e}".format(
  #     np.sum(
  #       np.linalg.norm(
  #         x_hist.reshape(step_num, -1) -
  #         (inputs.transpose(0, 3, 1, 2)).reshape(step_num, -1),
  #         axis=1
  #       )
  #     )
  #   )
  # )
  x_hist = jnp.where(jnp.abs(x_hist) < 5, x_hist, 5)

  # visualization
  if dim == 1:
    im_array = np.zeros((3, 1, *(outputs[..., 0].T).shape))
    im_array[0, 0] = inputs[..., 0].T
    im_array[1, 0] = x_hist[..., 0].T
    im_array[2, 0] = (inputs - x_hist)[..., 0].T
    plot_with_horizontal_colorbar(
      im_array, (12, 6), None, f"results/fig/{fig_name}.png", 100
    )
  elif dim == 2:
    n_plot = 6
    index_array = np.arange(
      0, n_plot * step_num // n_plot - 1, step_num // n_plot
    )
    im_array = np.zeros((3, n_plot, *(outputs[0, ..., 0].T).shape))
    for j in range(n_plot):
      im_array[0, j] = inputs[index_array[j], ..., 0]
      im_array[1, j] = x_hist[index_array[j], ..., 0]
      im_array[2, j] = (inputs - x_hist)[index_array[j], ..., 0]
    plot_with_horizontal_colorbar(
      im_array, (12, 6), None, f"results/fig/{fig_name}.png", 100
    )

  plot_stats_aux(
    np.arange(inputs.shape[0]) * model.dt,inputs[..., 0],
    x_hist[..., 0],f"results/fig/{fig_name}_stats.png",
  )


###############################################################################
#                   Numerical solver of the reaction-diffusion equation:
# For linear term, we use different discretization scheme, e.g. explicit,
#  implicit, Crank-Nielson, ADI etc.
# For nonlinear term, we use the explicit scheme (need to implement Ashford)
###############################################################################


def assembly_RDmatrix(n, dt, dx, beta=1.0, gamma=0.05, d=2):
  """assemble matrices used in the calculation
    A1 = I - gamma dt \Delta, used in implicit discretization of diffusion term, size n2*n2
    A2 = I - gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    A3 = I + gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    D, size 4n2*n2, Jacobi of the Newton solver in CN discretization
    :d: ratio between the diffusion coeff for u & v
    """

  global L_uminus, L_uplus, L_vminus, L_vplus, A_uplus, A_uminus, A_vplus, A_vminus, \
            A0_u, A0_v, L, L2

  L = jnp.eye(n) * -2 + jnp.eye(n, k=1) + jnp.eye(n, k=-1)
  L = L.at[0, -1].set(1)
  L = L.at[-1, 0].set(1)
  L = L / (dx**2)
  L2 = jnp.kron(L, jnp.eye(n)) + jnp.kron(jnp.eye(n), L)

  # matrix for ADI scheme
  L_uminus = jnp.eye(n) - L * gamma * dt / 2
  L_uplus = jnp.eye(n) + L * gamma * dt / 2
  L_vminus = jnp.eye(n) - L * gamma * dt / 2 * d
  L_vplus = jnp.eye(n) + L * gamma * dt / 2 * d

  A0_u = jnp.eye(n * n) + L2 * gamma * dt
  A0_v = jnp.eye(n * n) + L2 * gamma * dt * d
  A_uplus = jnp.eye(n * n) + L2 * gamma * dt / 2
  A_uminus = jnp.eye(n * n) - L2 * gamma * dt / 2
  A_vplus = jnp.eye(n * n) + L2 * gamma * dt / 2 * d
  A_vminus = jnp.eye(n * n) - L2 * gamma * dt / 2 * d
  # L = spa.csc_matrix(L)
  # L_uminus = spa.csc_matrix(L_uminus)
  # L_uplus = spa.csc_matrix(L_uplus)
  # L_vminus = spa.csc_matrix(L_vminus)
  # L_vplus = spa.csc_matrix(L_vplus)
  # L2 = spa.kron(L, np.eye(n)) + spa.kron(np.eye(n), L)
  # A_uplus = spa.eye(n*n) + L2 * gamma * dt/2
  # A_uminus = spa.eye(n*n) - L2 * gamma * dt/2
  # A_vplus = spa.eye(n*n) + L2 * gamma * dt/2 * d
  # A_vminus = spa.eye(n*n) - L2 * gamma * dt/2 * d


def RD_exp(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """
    explicit forward Euler solver for FitzHugh-Nagumo RD equation
    :input:
    u, v: initial condition, shape [nx, ny], different nx, ny is used for
    different diffusion coeff
    """

  nx = u.shape[0]
  ny = u.shape[1]
  u_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  v_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  rhsu_ = jnp.zeros([nx, ny])
  rhsv_ = jnp.zeros([nx, ny])
  if jnp.linalg.norm(source) != 0:
    rhsu_ = source[0].reshape(nx * ny)
    rhsv_ = source[1].reshape(nx * ny)
  u = u.reshape(nx * ny)
  v = v.reshape(nx * ny)

  for i in range(step_num):
    tmpu = A0_u @ u + dt * (u - v - u**3 + alpha) + rhsu_ * dt
    tmpv = A0_v @ v + beta * dt * (u - v) + rhsv_ * dt
    u = tmpu
    v = tmpv

    if (i + 1) % writeInterval == 0:
      u_hist = u_hist.at[(i - 0) // writeInterval].set(u.reshape(nx, ny))
      v_hist = v_hist.at[(i - 0) // writeInterval].set(v.reshape(nx, ny))

  return u_hist, v_hist


def RD_semi(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """semi-implicit solver for FitzHugh-Nagumo RD equation"""

  nx = u.shape[0]
  ny = u.shape[1]
  u_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  v_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  rhsu_ = jnp.zeros([nx, ny])
  rhsv_ = jnp.zeros([nx, ny])
  if jnp.linalg.norm(source) != 0:
    rhsu_ = source[0].reshape(nx * ny)
    rhsv_ = source[1].reshape(nx * ny)
  u = u.reshape(nx * ny)
  v = v.reshape(nx * ny)

  for i in range(step_num):
    rhsu = A_uplus @ u + dt * (u - v - u**3 + alpha) + rhsu_ * dt
    rhsv = A_vplus @ v + beta * dt * (u - v) + rhsv_ * dt
    u, _ = jsla.cg(A_uminus, rhsu)
    v, _ = jsla.cg(A_vminus, rhsv)

    if (i + 1) % writeInterval == 0:
      u_hist = u_hist.at[(i - 0) // writeInterval].set(u.reshape(nx, ny))
      v_hist = v_hist.at[(i - 0) // writeInterval].set(v.reshape(nx, ny))

  return u_hist, v_hist


def RD_adi(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """ADI solver for FitzHugh-Nagumo RD equation"""

  u = jnp.array(u)
  v = jnp.array(v)

  @jax.jit
  def update(u, v):

    rhsu = rhsu_ * dt + L_uplus @ u @ L_uplus + dt * (u - v - u**3 + alpha)
    rhsv = rhsv_ * dt + L_vplus @ v @ L_vplus + beta * dt * (u - v)
    return solve(L_uminus, solve(L_uminus, rhsu).T).T,\
      solve(L_uminus, solve(L_vminus, rhsv).T).T

  nx = u.shape[0]
  ny = u.shape[1]
  u_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  v_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  rhsu_ = jnp.zeros([nx, ny])
  rhsv_ = jnp.zeros([nx, ny])
  if jnp.linalg.norm(source) != 0:
    rhsu_ = source[0].reshape(nx, ny)
    rhsv_ = source[1].reshape(nx, ny)
  flag = True

  for i in range(step_num):
    u, v = update(u, v)
    if jnp.any(jnp.isnan(u)) or jnp.any(jnp.isnan(v)) or jnp.any(
      jnp.isinf(u)
    ) or jnp.any(jnp.isinf(v)):
      flag = False
      break

    if (i + 1) % writeInterval == 0:
      u_hist = u_hist.at[(i - 0) // writeInterval, :].set(u)
      v_hist = v_hist.at[(i - 0) // writeInterval, :].set(v)

  return u_hist, v_hist, flag


def RD_cn(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """full implicit solver with Crank-Nielson discretization
    TODO: we have not tested this function yet"""

  global L, D_
  dt = 1 / step_num
  t_array = np.array([5, 10, 20, 40, 80])

  #t_array = np.array([1, 2, 3, 4, 80])

  #plt.subplot(231)
  #plt.imshow(u.reshape(n, n), cmap = cm.viridis)

  def F(u_next, v_next, u, v):
    Fu = A2 @ u_next - A3 @ u + (
      u_next**3 + u**3 + v_next + v - u_next - u - alpha
    ) * dt / 2
    Fv = A2 @ v_next - A3 @ v + (v_next + v - u_next - u) * dt * beta / 2
    res = np.hstack([Fu, Fv])
    return res

  def Newton(n):

    global u, v, L, D_
    # we use the semi-implicit scheme iteration as the initial guess of Newton method
    rhsu = u + dt * (u - v + u**3 + alpha)
    rhsv = v + beta * dt * (u - v)
    u_next = sps(A1, rhsu)
    v_next = sps(A1, rhsv)
    res = F(u_next, v_next, u, v)

    count = 0
    while jnp.linalg.norm(res) > tol:
      D_[:n * n, :n *
         n] = D_[:n * n, :n *
                 n] + dt / 2 * (spa.diags(3 * (u_next**2)) - spa.eye(n * n))
      D = D_.tocsr()
      #duv = sps(D, res)
      # GMRES with initial guess pressure in last time step
      duv = jsla.gmres(A=D, b=res.reshape(n * n), x0=duv).reshape([n, n])
      # BiSTABCG with initial guess pressure in last time step
      duv = jsla.bicgstab(A=D, b=res.reshape(n * n), x0=duv).reshape([n, n])
      u_next = u_next - duv[:n * n]
      v_next = v_next - duv[n * n:]
      res = F(u_next, v_next, u, v)
      count = count + 1
      print(scalg.norm(res))
    print(count)

    u = u_next
    v = v_next

  for i in range(step_num):
    for j in range(5):
      #if i == t_array[j] * step_num / 100:
      #    plt.subplot(2, 3, j+2)
      #    plt.imshow(u.reshape(n, n), cmap = cm.viridis)
      #    plt.colorbar()
      Newton()

  #plt.show()
  return u, v


def assembly_NSmatrix(nx, ny, dx, dy, BC: str = "Dirichlet"):
  """assemble matrices used in the calculation
    LD: Laplacian operator with Dirichlet BC
    LN: Laplacian operator with Neuman BC, notice that this operator may have
    different form depends on the position of the boundary, here we use the
    case that boundary is between the outmost two grids
    L:  Laplacian operator associated with current BC with three Neuman BCs on
    upper, lower, left boundary and a Dirichlet BC on right
    """

  global L

  def Laplacian_Neumann(n):
    LN = jnp.roll(jnp.eye(n), 1, axis=1) + jnp.roll(jnp.eye(n), -1, axis=1) -\
      jnp.eye(n) * 2
    LN = LN.at[0, 0].set(-1)
    LN = LN.at[0, -1].set(0)
    LN = LN.at[-1, -1].set(-1)
    LN = LN.at[-1, 0].set(0)
    return LN

  LNx = Laplacian_Neumann(nx)
  LNy = Laplacian_Neumann(ny)
  L = jnp.kron(LNx /
               (dx**2), jnp.eye(ny)) + jnp.kron(jnp.eye(nx), LNy / (dy**2))
  if BC == "Dirichlet":
    for i in range(ny):
      L = L.at[-1 - i, -1 - i].add(-2 / (dx**2))
  elif BC == "Neumann":
    L = jnp.vstack([L, jnp.ones_like(L[0:1])])
    L = jnp.hstack([L, jnp.ones_like(L[:, 0:1])])
    L = L.at[-1, -1].set(0)


def projection_correction(
  u: jnp.ndarray,
  v: jnp.ndarray,
  p: jnp.ndarray,
  t: float,
  dx=1 / 32,
  dy=1 / 32,
  nx=128,
  ny=32,
  y0=0.325,
  dt=.01,
  Re=100,
  BC: str = "Dirichlet",
  correction: bool = False,
):
  """projection method to solve the incompressible NS equation
    The convection discretization is given by central difference
    u_ij (u_i+1,j - u_i-1,j)/2dx + \Sigma v_ij (u_i,j+1 - u_i,j-1)/2dx
    
  The collocation point of p locates at the center of the cell (1/2, 1/2)
  The collocation point of u locates at the right of p (1, 1/2)
  The collocation point of v locates at the top of p (1/2, 1)
  v[:, -1] = 0 for the no-slip boundary condition

  """

  def _u_padx(u: jnp.ndarray):
    return jnp.vstack([u_inlet, u, u[-1]])

  def _v_padx(v: jnp.ndarray):
    return jnp.vstack([2 * v_inlet - v[0], v, v[-1]])

  def _u_pady(u: jnp.ndarray):
    return jnp.hstack([-u[:, 0:1], u, -u[:, -1:]])

  def _v_pady(v: jnp.ndarray):
    return jnp.hstack([jnp.zeros_like(v[:, 0:1]), v, -v[:, -2:-1]])

  def _v2u(v: jnp.ndarray):
    """interpolate v to u"""
    v_pady = jnp.hstack([jnp.zeros_like(v[:, 0:1]), v])
    v = jnp.vstack([v_pady, v_pady[-1]])
    return (v[1:, 1:] + v[:-1, 1:] + v[1:, :-1] + v[:-1, :-1]) / 4

  def _u2v(u: jnp.ndarray):
    """interpolate u to v"""
    u_padx = jnp.vstack([u_inlet, u])
    u = jnp.hstack([u_padx, -u_padx[:, -1:]])
    return (u[1:, 1:] + u[:-1, 1:] + u[1:, :-1] + u[:-1, :-1]) / 4

  def grad_p(p: jnp.ndarray, dpdn: float = 0):
    """calculate the gradient of the pressure
    
    p: (nx, ny)
    dpdx: (nx - 1, ny)
    dpdy: (nx, ny - 1)
    dpdx[-1] = dpdy[:, -1] = 0 since p satisfies the Neuman BC 
    """
    if BC == "Dirichlet":
      p_padx = jnp.vstack([p, -p[-1:]])
    elif BC == "Neumann":
      p_padx = jnp.vstack([p, p[-1:] + dpdn * dx])
    dpdx = (p_padx[1:] - p_padx[:-1]) / dx
    dpdy = (p[:, 1:] - p[:, :-1]) / dy
    return dpdx, dpdy

  def div_uv(u: jnp.ndarray, v: jnp.ndarray):
    """calculate the divergence of the velocity field"""
    u_padx = jnp.vstack([u_inlet, u])
    dudx = (u_padx[1:] - u_padx[:-1]) / dx
    v_pady = jnp.hstack([jnp.zeros_like(v[:, 0:1]), v])
    dvdy = (v_pady[:, 1:] - v_pady[:, :-1]) / dy
    return dudx + dvdy

  def laplace_uv(u: jnp.ndarray, v: jnp.ndarray):
    """calculate the Laplacian of the velocity field"""

    u_padx = _u_padx(u)
    u_pad = _u_pady(u_padx)
    lapl_u = (u_pad[2:, 1:-1] - 2 * u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1]) / dx**2 +\
      (u_pad[1:-1, 2:] - 2 * u_pad[1:-1, 1:-1] + u_pad[1:-1, :-2]) / dy**2
    v_padx = _v_padx(v)
    v_pad = _v_pady(v_padx)
    lapl_v = (v_pad[1:-1, 2:] - 2 * v_pad[1:-1, 1:-1] + v_pad[1:-1, :-2]) / dy**2 +\
      (v_pad[2:, 1:-1] - 2 * v_pad[1:-1, 1:-1] + v_pad[:-2, 1:-1]) / dx**2
    return lapl_u, lapl_v

  def transport(u: jnp.ndarray, v: jnp.ndarray):
    """calculate the transport term of the velocity field"""
    u_padx = _u_padx(u)
    u_pady = _u_pady(u)
    uu_x = (u_padx[2:] - u_padx[:-2]) / dx / 2 * u
    vu_y = (u_pady[:, 2:] - u_pady[:, :-2]) / dy / 2 * _v2u(v)
    v_padx = _v_padx(v)
    v_pady = _v_pady(v)
    uv_x = (v_padx[2:] - v_padx[:-2]) / dx / 2 * _u2v(u)
    vv_y = (v_pady[:, 2:] - v_pady[:, :-2]) / dy / 2 * v

    return uu_x + vu_y, uv_x + vv_y

  def inlet(y: jnp.ndarray):
    """set the inlet velocity"""
    return y * (1 - y) * jnp.exp(-10 * (y - y0)**2) * 3

  u_inlet = inlet(np.linspace(dy / 2, 1 - dy / 2, ny))
  v_inlet = inlet(np.linspace(dy, 1, ny)) * jnp.sin(t)

  dpdx, dpdy = grad_p(p)
  lapl_u, lapl_v = laplace_uv(u, v)
  du, dv = transport(u, v)
  u += -dpdx * dt
  v = v.at[:, :-1].add(-dpdy * dt)
  u += dt * (lapl_u / Re - du)
  v += dt * (lapl_v / Re - dv)
  v = v.at[:, -1].set(0)

  if not correction:
    # pressure correction
    res = div_uv(u, v) / dt
    if BC == "Dirichlet":
      p_res = jnp.linalg.solve(L, res.reshape(-1)).reshape([nx, ny])
      dpdn = 0
    elif BC == "Neumann":
      dpdn = -res.sum() / ny * dx
      res = res.at[-1].add(dpdn / dx)
      p_res = jnp.linalg.solve(L, jnp.hstack([res.reshape(-1),
                                              jnp.zeros(1)]
                                             ))[:-1].reshape([nx, ny])

    dpdx, dpdy = grad_p(p_res, -dpdn)
    u += -dpdx * dt
    v = v.at[:, :-1].add(-dpdy * dt)
    p += p_res

  res_ = div_uv(u, v)
  if jnp.linalg.norm(res_) > 1e-12:
    print(jnp.linalg.norm(res_))
    print("Velocity field is not divergence free!!!")

  return u, v, p


def a_posteriori_analysis(
  config: ml_collections.ConfigDict,
  ks_fine,
  ks_coarse,
  correction_nn: callable,
  params,
):

  c = config.sim.c
  L = config.sim.L
  T = config.sim.T
  init_scale = config.sim.init_scale
  BC = config.sim.BC
  # solver parameters
  if BC == "periodic":
    N1 = config.sim.n
  elif BC == "Dirichlet-Neumann":
    N1 = config.sim.n - 1
  r = config.sim.r
  N2 = N1 // r
  rng = random.PRNGKey(config.sim.seed)
  train_mode = config.train.mode
  n_g = config.train.n_g

  # a posteriori analysis
  rng, key = random.split(rng)
  if BC == "periodic":
    u0 = ks_fine.attractor + init_scale * random.normal(key) *\
      jnp.sin(10 * jnp.pi * jnp.linspace(0, L - L/N1, N1) / L)
  elif BC == "Dirichlet-Neumann":
    dx = L / (N1 + 1)
    x = jnp.linspace(dx, L - dx, N1)
    # u0 = ks_fine.attractor + init_scale * random.normal(key) *\
    #   jnp.sin(10 * jnp.pi * x / L)
    # u0 = random.uniform(key) * jnp.sin(8 * jnp.pi * x / 128) +\
    #   random.uniform(rng) * jnp.sin(16 * jnp.pi * x / 128)
    r0 = random.uniform(key) * 20 + 44
    u0 = jnp.exp(-(x - r0)**2 / r0**2 * 4)
  ks_fine.run_simulation(u0, ks_fine.CN_FEM)
  if config.test.solver == "CN":
    ks_coarse.run_simulation(u0[r - 1::r], ks_coarse.CN_FEM)
  elif config.test.solver == "RK4":
    ks_coarse.run_simulation(u0[r - 1::r], ks_coarse.RK4)
  # im_array = jnp.zeros(
  #   (3, 1, ks_coarse.x_hist.shape[1], ks_coarse.x_hist.shape[0])
  # )
  # im_array = im_array.at[0, 0].set(ks_fine.x_hist[:, r-1::r].T)
  # im_array = im_array.at[1, 0].set(ks_coarse.x_hist.T)
  # im_array = im_array.at[2,
  #                        0].set(ks_coarse.x_hist.T - ks_fine.x_hist[:, r-1::r].T)
  # title_array = [f"{N1}", f"{N2}", "diff"]
  baseline = ks_coarse.x_hist

  if train_mode == "regression":

    def corrector(input):
      """
      input.shape = (N, )
      output.shape = (N, )
      """

      dx = L / N2
      u_x = (jnp.roll(input, 1) - jnp.roll(input, -1)) / dx / 2
      u_xx = ((jnp.roll(input, 1) + jnp.roll(input, -1)) - 2 * input) / dx**2
      u_xxxx = ((jnp.roll(input, 2) + jnp.roll(input, -2)) -\
          4*(jnp.roll(input, 1) + jnp.roll(input, -1)) + 6 * input) /\
          dx**4
      # local model: [1] to [1]
      if config.train.input == "u":
        return partial(correction_nn.apply,
                       params)(input.reshape(-1, 1)).reshape(-1)
      elif config.train.input == "ux":
        return partial(correction_nn.apply, params)(u_x.reshape(-1,
                                                                1)).reshape(-1)
      elif config.train.input == "uxx":
        return partial(correction_nn.apply,
                       params)(u_xx.reshape(-1, 1)).reshape(-1)
      elif config.train.input == "uxxxx":
        return partial(correction_nn.apply,
                       params)(u_xxxx.reshape(-1, 1)).reshape(-1)
      elif config.train.input == "uglobal":
        return partial(correction_nn.apply,
                       params)(input.reshape(1, -1)).reshape(-1)
      # global model: [N] to [N]
      # return partial(correction_nn.apply,
      #                params)(input.reshape(1, -1)).reshape(-1)

  elif train_mode == "generative":
    vae_bind = vae.bind({"params": params})
    z = random.normal(key, shape=(1, config.train.vae.latents))
    corrector = partial(vae_bind.generate, z)
  elif train_mode == "gaussian":
    z = random.normal(key)

    def corrector(input: jnp.array):
      """
      input.shape = (N, )
      output.shape = (N, )
      """

      dx = L / N2
      u_x = (jnp.roll(input, 1) - jnp.roll(input, -1)) / dx / 2
      u_xx = ((jnp.roll(input, 1) + jnp.roll(input, -1)) - 2 * input) / dx**2
      u_xxxx = ((jnp.roll(input, 2) + jnp.roll(input, -2)) -\
          4*(jnp.roll(input, 1) + jnp.roll(input, -1)) + 6 * input) /\
          dx**4
      # local model: [1] to [1]
      if config.train.input == "u":
        tmp = partial(correction_nn.apply, params)(input.reshape(-1, 1))
      elif config.train.input == "ux":
        tmp = partial(correction_nn.apply, params)(u_x.reshape(-1, 1))
      elif config.train.input == "uxx":
        tmp = partial(correction_nn.apply, params)(u_xx.reshape(-1, 1))
      elif config.train.input == "uxxxx":
        tmp = partial(correction_nn.apply, params)(u_xxxx.reshape(-1, 1))
      p = jax.nn.sigmoid(tmp[..., :n_g])
      index = jax.vmap(partial(jax.random.choice, key=key, a=n_g,
                               shape=(1, )))(p=p).reshape(-1)
      return (
        tmp[np.arange(N2), n_g + index] + tmp[np.arange(N2), n_g + index] * z
      ).reshape(-1)

    def corrector_sample(input: jnp.array, rng: PRNGKey):

      dx = L / N2
      u_x = (jnp.roll(input, 1) - jnp.roll(input, -1)) / dx / 2
      u_xx = ((jnp.roll(input, 1) + jnp.roll(input, -1)) - 2 * input) / dx**2
      u_xxxx = ((jnp.roll(input, 2) + jnp.roll(input, -2)) -\
          4*(jnp.roll(input, 1) + jnp.roll(input, -1)) + 6 * input) /\
          dx**4
      # local model: [1] to [1]
      if config.train.input == "u":
        tmp = partial(correction_nn.apply, params)(input.reshape(-1, 1))
      elif config.train.input == "ux":
        tmp = partial(correction_nn.apply, params)(u_x.reshape(-1, 1))
      elif config.train.input == "uxx":
        tmp = partial(correction_nn.apply, params)(u_xx.reshape(-1, 1))
      elif config.train.input == "uxxxx":
        tmp = partial(correction_nn.apply, params)(u_xxxx.reshape(-1, 1))
      p = jax.nn.sigmoid(tmp[..., :n_g])
      index = jax.vmap(partial(jax.random.choice, key=key, a=n_g,
                               shape=(1, )))(p=p).reshape(-1)
      z = random.normal(key)
      return (
        tmp[np.arange(N2), n_g + index] + tmp[np.arange(N2), n_g + index] * z
      ).reshape(-1)

  if config.test.solver == "CN":
    ks_coarse.run_simulation_with_correction(
      u0[r - 1::r], ks_coarse.CN_FEM, corrector
    )
  elif config.test.solver == "RK4":
    ks_coarse.run_simulation_with_correction(
      u0[r - 1::r], ks_coarse.RK4, corrector
    )
  correction1 = ks_coarse.x_hist
  correction2 = None
  if train_mode == "gaussian":
    if config.test.solver == "CN":
      ks_coarse.run_simulation_with_probabilistic_correction(
        u0[r - 1::r], ks_coarse.CN_FEM, corrector_sample
      )
    elif config.test.solver == "RK4":
      ks_coarse.run_simulation_with_probabilistic_correction(
        u0[r - 1::r], ks_coarse.RK4, corrector_sample
      )
    correction2 = ks_coarse.x_hist

  # compare the simulation statistics (A posteriori analysis)
  from ml4dynamics.visualize import plot_stats
  plot_stats(
    np.arange(ks_fine.x_hist.shape[0]) * ks_fine.dt,
    ks_fine.x_hist,
    baseline,
    correction1,
    correction2,
    f"results/fig/ks_c{c}T{T}n{config.sim.case_num}_{train_mode}_stats.png",
  )
  # plot_with_horizontal_colorbar(
  #   im_array,
  #   fig_size=(4, 6),
  #   title_array=title_array,
  #   file_path=
  #   f"results/fig/ks_nu{nu}_N{N1}n{config.sim.case_num}_{train_mode}_cmp.pdf"
  # )
  # im_array = jnp.zeros(
  #   (3, 1, ks_coarse.x_hist.shape[1], ks_coarse.x_hist.shape[0])
  # )
  # im_array = im_array.at[0, 0].set(ks_fine.x_hist[:, r-1::r].T)
  # im_array = im_array.at[1, 0].set(ks_coarse.x_hist.T)
  # im_array = im_array.at[2,
  #                        0].set(ks_coarse.x_hist.T - ks_fine.x_hist[:, r-1::r].T)
  # plot_with_horizontal_colorbar(
  #   im_array,
  #   fig_size=(4, 6),
  #   title_array=title_array,
  #   file_path=
  #   f"results/fig/ks_nu{nu}_N{N1}n{config.sim.case_num}_{train_mode}_correct_cmp.pdf"
  # )


def plot_with_horizontal_colorbar(
  im_array, fig_size=(10, 4), title_array=None, file_path=None, dpi=100
):

  if not isinstance(im_array, np.ndarray):
    im_array = np.array(im_array)
  fig, axs = plt.subplots(
    im_array.shape[0], im_array.shape[1], figsize=fig_size
  )
  axs = axs.flatten()
  fraction = 0.05
  pad = 0.001
  for i in range(im_array.shape[0]):
    for j in range(im_array.shape[1]):
      im = axs[i * im_array.shape[1] +
               j].imshow(im_array[i, j], cmap=cm.twilight)
      if title_array is not None and\
        title_array[i * im_array.shape[1] +j] is not None:
        axs[i * im_array.shape[1] +
            j].set_title(title_array[i * im_array.shape[1] + j])
      axs[i * im_array.shape[1] + j].axis("off")
      fig.colorbar(
        im,
        ax=axs[i * im_array.shape[1] + j],
        fraction=fraction,
        pad=pad,
        orientation="horizontal"
      )
  fig.tight_layout(pad=0.0)

  if file_path is not None:
    plt.savefig(file_path, dpi=dpi)
  plt.close()


def jax_memory_profiler(verbose: bool = False):

  all_objects = gc.get_objects()
  total_size = 0
  aux_size = 0
  for obj in all_objects:
    if isinstance(obj, jnp.ndarray):
      total_size += jax.device_get(obj).nbytes
      if jax.device_get(obj).nbytes > 1e7 and\
        'nvidia' in obj.device.device_kind.lower():
        aux_size += jax.device_get(obj).nbytes
        if verbose:
          print(obj.dtype)
          print(obj.shape)
          print(jax.device_get(obj).nbytes)
  print(f"total_size of large array on gpu: {aux_size/1e9:.3f}GB")
  print(f"total_size: {total_size/1e9:.3f}GB")
