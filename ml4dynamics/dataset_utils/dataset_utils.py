import jax
import jax.numpy as jnp


def calc_correction(rd_fine, rd_coarse, nx: float, r: int, uv: jnp.ndarray):
  next_step_fine = rd_fine.adi(uv.reshape(-1)).reshape(2, nx, nx)
  tmp = (
    uv[:, 0::2, 0::2] + uv[:, 1::2, 0::2] +
    uv[:, 0::2, 1::2] + uv[:, 1::2, 1::2]
  ) / 4
  uv_ = tmp.reshape(-1)
  next_step_coarse = rd_coarse.adi(uv_).reshape(2, nx // r, nx // r)
  next_steo_coarse_interp = jnp.vstack(
    [
      jnp.kron(next_step_coarse[0], jnp.ones((r, r))).reshape(1, nx, nx),
      jnp.kron(next_step_coarse[1], jnp.ones((r, r))).reshape(1, nx, nx),
    ]
  )
  return next_step_fine - next_steo_coarse_interp