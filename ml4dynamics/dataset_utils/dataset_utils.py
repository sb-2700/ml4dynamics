import jax
import jax.numpy as jnp
import ml_collections
from box import Box


def _create_box_filter(N1, N2, r, BC):
  """Create box filter (averaging) operator"""
  res_op = jnp.zeros((N2, N1))
  
  if r == 2:
    raise Exception("Deprecated...")
    res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r].set(1)
    res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r + 2].set(1)
  elif r == 4:
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
  return res_op


def _create_gaussian_filter(N1, N2, r, BC):
  """Create Gaussian filter operator using same edge logic as box filter"""
  res_op = jnp.zeros((N2, N1))
  
  # Gaussian kernel width
  sigma = r / 3.0
  
  # Use same stencil size as box filter
  stencil_size = 7
  
  for i in range(N2):
    # Same logic as box filter: start at i * r
    start_idx = i * r
    end_idx = min(start_idx + stencil_size, N1)  # Don't go beyond array bounds
    
    # Create Gaussian weights for this stencil
    center = start_idx + stencil_size // 2  # Center of stencil
    for j in range(start_idx, end_idx):
      distance = abs(j - center)
      weight = jnp.exp(-0.5 * (distance / sigma)**2)
      res_op = res_op.at[i, j].set(weight)
  
  # Handle periodic BC same way as box filter
  if BC == "periodic":
    # Last row wraps around (same as box filter)
    i = N2 - 1
    for j in range(3):  # First 3 points, same as box filter
      distance = abs(j - (stencil_size // 2))  # Distance from center of stencil
      weight = jnp.exp(-0.5 * (distance / sigma)**2)
      res_op = res_op.at[i, j].set(weight)
  
  # Normalize each row to sum to 1, with safety check
  row_sums = jnp.sum(res_op, axis=1, keepdims=True)
  # Add small epsilon to avoid division by very small numbers
  res_op = res_op / (row_sums + 1e-12)
  return res_op


def _create_spectral_filter(N1, N2, r, BC):
  """Create spectral cutoff filter operator"""
  if BC == "Dirichlet-Neumann":
    # For non-periodic BC, use DCT-based spectral filter
    # This is more complex - for now, fall back to a smooth approximation
    return _create_smooth_spectral_filter_nonperiodic(N1, N2, r)
  else:
    # For periodic BC, use FFT-based spectral filter
    return _create_spectral_filter_periodic(N1, N2, r)


def _create_spectral_filter_periodic(N1, N2, r):
  """FFT-based spectral filter for periodic BC"""
  # Create filter in frequency domain then transform to spatial
  cutoff_freq = N2 // 2  # Cutoff frequency for coarse grid
  
  res_op = jnp.zeros((N2, N1))
  for i in range(N2):
    # Create impulse at coarse grid location
    impulse = jnp.zeros(N1)
    impulse = impulse.at[i * r].set(1.0)
    
    # Apply spectral filter in frequency domain
    impulse_hat = jnp.fft.fft(impulse)
    # Low-pass filter: keep low frequencies, zero high frequencies
    filtered_hat = jnp.where(
      jnp.abs(jnp.fft.fftfreq(N1) * N1) <= cutoff_freq,
      impulse_hat, 0.0
    )
    filtered = jnp.real(jnp.fft.ifft(filtered_hat))
    res_op = res_op.at[i, :].set(filtered)
  
  # Normalize
  row_sums = jnp.sum(res_op, axis=1, keepdims=True)
  res_op = res_op / row_sums
  return res_op


def _create_smooth_spectral_filter_nonperiodic(N1, N2, r):
  """Smooth spectral-like filter for non-periodic BC"""
  res_op = jnp.zeros((N2, N1))
  
  # Vectorized computation
  i_indices = jnp.arange(N2)
  j_indices = jnp.arange(N1)
  
  # Create meshgrid for vectorized computation
  I, J = jnp.meshgrid(i_indices, j_indices, indexing='ij')
  
  # Centers of coarse grid cells
  centers = I * r + r // 2
  distances = jnp.abs(J - centers)
  
  # Smooth cutoff function (approximates sinc-like behavior)
  weights = jnp.where(
    distances <= r,
    0.5 * (1 + jnp.cos(jnp.pi * distances / r)),
    0.0
  )
  
  # Normalize each row to sum to 1
  row_sums = jnp.sum(weights, axis=1, keepdims=True)
  res_op = weights / row_sums
  return res_op


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
    ],
    axis=2
  )

  return next_step_fine - next_step_coarse_interp


def res_int_fn(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  r = config.sim.rx
  if config.case == "ks":
    BC = config.sim.BC
    filter_type = config.sim.get('filter_type', 'box')  # default to box filter
    if BC == "periodic":
      N1 = config.sim.n
    elif BC == "Dirichlet-Neumann":
      N1 = config.sim.n - 1
    N2 = N1 // r
    
    res_op = jnp.zeros((N2, N1))
    int_op = jnp.zeros((N1, N2))
    
    if filter_type == "box":
      # Original box filter implementation
      res_op = _create_box_filter(N1, N2, r, BC)
    elif filter_type == "gaussian":
      res_op = _create_gaussian_filter(N1, N2, r, BC)
    elif filter_type == "spectral":
      res_op = _create_spectral_filter(N1, N2, r, BC)
    else:
      raise ValueError(f"Unknown filter_type: {filter_type}")
    
    int_op = jnp.linalg.pinv(res_op)
    assert jnp.allclose(res_op @ int_op, jnp.eye(N2), atol=1e-3)
    assert jnp.allclose(res_op.sum(axis=-1), jnp.ones(N2), atol=1e-6)

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
        ],
        axis=2
      )

  return res_fn, int_fn
