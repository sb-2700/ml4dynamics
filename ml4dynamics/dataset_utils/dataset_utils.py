import jax.numpy as jnp


def calc_correction(rd_fine, rd_coarse, nx: float, r: int, uv: jnp.ndarray):
  """
  Args:
    uv: shape = [2, nx, nx]
  """
  next_step_fine = rd_fine.adi(uv.reshape(-1)).reshape(2, nx, nx)
  tmp = jnp.zeros((2, nx // r, nx // r))
  for k in range(r):
    for j in range(r):
      tmp += uv[:, k::r, j::r]
  tmp = tmp / (r**2)
  next_step_coarse = rd_coarse.adi(tmp.reshape(-1)).reshape(2, nx // r, nx // r)
  next_step_coarse_interp = jnp.vstack(
    [
      jnp.kron(next_step_coarse[0], jnp.ones((r, r))).reshape(1, nx, nx),
      jnp.kron(next_step_coarse[1], jnp.ones((r, r))).reshape(1, nx, nx),
    ]
  )
  return next_step_fine - next_step_coarse_interp