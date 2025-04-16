import jax
import jax.numpy as jnp
import numpy as np
import torch
from dlpack import asdlpack


def run_simulation_coarse_grid_correction(
  train_state, coarse_model, label: jnp.ndarray, r: int,
  beta: float, uv: jnp.ndarray
):
  r"""
  TODO: Come up with a better name, this is more or less a synthetic
  scenario.

  The simulation is performed over the fine grid $2n$, but in each step instead
  of advancing using the expensive fine-grid solver, we:
    1. restrict the fine-grid variable onto coarse grid;
    2. advance the coarse-grid variable using cheap coarse-grid solver;
    3. interpolate the updated coarse-grid variable to onto the fine grid;
    4. add a correction to it

  $$
    u_{k+1}^{2n} = I_n^{2n} \circ f_n(R_{2n}^{n}(u_k^{2n})) + y_k^{2n}
  $$

  Following the DAgger paper, we use beta to mix the current policy with the
  expert policy to stabilize the trajectory
  
  """

  @jax.jit
  def iter(uv: jnp.array, expert: jnp.array = 0):
    correction, _ = train_state.apply_fn_with_bn(
      {
        "params": train_state.params,
        "batch_stats": train_state.batch_stats
      },
      uv.reshape(1, *uv.shape),
      is_training=False
    )
    correction = correction.reshape(nx, nx, -1)
    tmp = jnp.zeros((nx // r, nx // r, 2))
    for k in range(r):
      for j in range(r):
        tmp += uv[k::r, j::r]
    tmp = tmp / (r**2)
    uv = coarse_model.adi(
      tmp.transpose(2, 0, 1).reshape(-1)
    ).reshape(2, nx // r, nx // r)
    uv = jnp.vstack(
      [
        jnp.kron(uv[0], jnp.ones((r, r))).reshape(1, nx, nx),
        jnp.kron(uv[1], jnp.ones((r, r))).reshape(1, nx, nx),
      ]
    ).reshape(nx, nx, 2)
    return uv + (correction * (1 - beta) + beta * expert) * dt

  nx = uv.shape[1]
  dt = coarse_model.dt
  step_num = coarse_model.step_num
  x_hist = np.zeros([step_num, nx, nx, 2])
  for i in range(step_num):
    x_hist[i] = uv
    uv = iter(uv, label[i])

  return x_hist


def run_simulation_coarse_grid_correction_torch(
  model, rd_fine, coarse_model, r: int, device, uv: jnp.ndarray
):

  nx = uv.shape[1]
  dt = coarse_model.dt
  step_num = rd_fine.step_num
  x_hist = jnp.zeros([step_num, 2, nx, nx])
  for i in range(step_num):
    x_hist = x_hist.at[i].set(uv)
    tmp = (
      uv[:, 0::2, 0::2] + uv[:, 1::2, 0::2] + uv[:, 0::2, 1::2] +
      uv[:, 1::2, 1::2]
    ) / 4
    uv = coarse_model.adi(tmp.reshape(-1)).reshape(2, nx // r, nx // r)
    uv = np.vstack(
      [
        np.kron(uv[0], np.ones((r, r))).reshape(1, nx, nx),
        np.kron(uv[1], np.ones((r, r))).reshape(1, nx, nx),
      ]
    )

    # naive jax-torch data exchange from numpy, gpu-cpu
    # uv_np = np.asarray(uv)
    # uv_torch = torch.from_numpy(uv_np).clone().to(device)
    # correction = model(uv_torch.reshape((1, *uv.shape)))
    # uv += jnp.array((correction[0].detach().cpu().numpy())) * dt

    # jax-torch data exchange via dlpack
    # reference: https://github.com/jax-ml/jax/issues/1100
    correction = model(
      torch.from_dlpack(asdlpack(uv)).reshape((1, *uv.shape)).to(device)
    )
    uv += jnp.array(jnp.from_dlpack(asdlpack(correction[0].detach()))) * dt

  return x_hist


def run_simulation_sgs(
  train_state, coarse_model, label: jnp.ndarray, beta: float, uv: jnp.ndarray
):

  @jax.jit
  def iter(uv: jnp.array, expert: jnp.array = 0):
    correction, _ = train_state.apply_fn_with_bn(
      {
        "params": train_state.params,
        "batch_stats": train_state.batch_stats
      },
      uv[None],
      is_training=False
    )
    uv = coarse_model.adi(
      uv.transpose(2, 0, 1).reshape(-1)
    ).reshape(2, nx, nx).transpose(1, 2, 0)
    return uv + (correction[0] * (1 - beta) + beta * expert) * dt * dx**2

  nx = uv.shape[1]
  dt = coarse_model.dt
  dx = coarse_model.dx
  step_num = coarse_model.step_num
  x_hist = np.zeros([step_num, nx, nx, 2])
  for i in range(step_num):
    x_hist[i] = uv
    uv = iter(uv, label[i])

  return x_hist


def run_ns_simulation_pressue_correction(
  train_state, ns_model, label: jnp.ndarray, beta: float, uv: jnp.ndarray
):
  
  uv = jnp.array(uv)
  nx, ny = uv.shape[:-1]
  step_num = ns_model.step_num
  x_hist = np.zeros([step_num, nx, ny, 2])
  for i in range(step_num):
    x_hist[i] = uv
    correction, _ = train_state.apply_fn_with_bn(
      {
        "params": train_state.params,
        "batch_stats": train_state.batch_stats
      },
      uv.reshape(1, *uv.shape),
      is_training=False
    )
    u, v, _ = ns_model.projection_correction(
      uv[..., 0], uv[..., 1], correction[0, ..., 0]
    )
    uv = jnp.concatenate([u[..., None], v[..., None]], axis=-1)

  return x_hist
