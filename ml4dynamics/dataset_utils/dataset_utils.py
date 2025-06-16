import jax
import jax.numpy as jnp
import ml_collections
from box import Box


def calc_correction(rd_fine, rd_coarse, nx: float, r: int, uv: jnp.ndarray):
  """
  Args:
    uv: shape = [nx, nx, 2]
  """
  next_step_fine = rd_fine.adi(uv)
  tmp = jnp.zeros((nx // r, nx // r, 2))
  for k in range(r):
    for j in range(r):
      tmp += uv[k::r, j::r]
  tmp = tmp / (r**2)
  next_step_coarse = rd_coarse.adi(tmp)
  next_step_coarse_interp = jnp.concatenate(
    [
      jnp.kron(next_step_coarse[..., 0], jnp.ones((r, r)))[..., None],
      jnp.kron(next_step_coarse[..., 1], jnp.ones((r, r)))[..., None],
    ], axis=2
  )

  return next_step_fine - next_step_coarse_interp


def res_int_fn(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  r = config.sim.rx
  if config.case == "ks":
    BC = config.sim.BC
    if BC == "periodic":
      N1 = config.sim.n
    elif BC == "Dirichlet-Neumann":
      N1 = config.sim.n - 1
    N2 = N1 // r
    res_op = jnp.zeros((N2, N1))
    int_op = jnp.zeros((N1, N2))
    if r == 2:
      raise Exception("Deprecated...")
      res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r].set(1)
      res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r + 2].set(1)
    elif r == 4:
      # stencil = 4
      # for i in range(N2):
      #   res_op = res_op.at[i, i * r + 1:i * r + 6].set(1)
      # res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r + 3].set(0)
      # if BC == "periodic":
      #   res_op = res_op.at[-1, :r // 2].set(1)

      # stencil = 7
      for i in range(N2):
        res_op = res_op.at[i, i * r:i * r + 7].set(1)
      if BC == "periodic":
        res_op = res_op.at[-1, :3].set(1)
      res_op /= 7 / 4
    elif r == 8:
      # stencil = 12
      for i in range(N2):
        res_op = res_op.at[i, i * r + 3:i * r + 12].set(1)
      res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r + 7].set(0)
    res_op /= r
    int_op = jnp.linalg.pinv(res_op)
    assert jnp.allclose(res_op @ int_op, jnp.eye(N2))
    assert jnp.allclose(res_op.sum(axis=-1), jnp.ones(N2))
    @jax.jit
    def res_fn(x):
      return (x.reshape(-1, N1) @ res_op.T).reshape(N2, -1)
    @jax.jit
    def int_fn(x):
      return (x.reshape(-1, N2) @ int_op.T).reshape(N1, -1)
  elif config.case == "react_diff" or config.case == "ns_hit":
    n = config.sim.n
    @jax.jit
    def res_fn(x):
      result = jnp.zeros((n // r, n // r, x.shape[-1]))
      for k in range(r):
        for j in range(r):
          result += x[k::r, j::r]
      return result / (r**2)

    @jax.jit
    def int_fn(x):
      # only works for the case with 2 components
      return jnp.concatenate(
        [
          jnp.kron(x[..., 0], jnp.ones((r, r)))[..., None],
          jnp.kron(x[..., 1], jnp.ones((r, r)))[..., None],
        ], axis=2
      )
  return res_fn, int_fn
