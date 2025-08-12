import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter1d
import ml_collections
from box import Box
from scipy.signal import resample

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


def _create_box_filter(N1, N2, r, BC, s):
  """Create box filter (averaging) operator
  
  Args:
    N1: fine grid size
    N2: coarse grid size  
    r: coarsening ratio
    BC: boundary condition ("periodic" or "Dirichlet-Neumann")
    s: stencil size (must be odd)
  """
  if s % 2 == 0:
    raise ValueError("Stencil size must be odd")
    
  res_op = jnp.zeros((N2, N1))
  half_stencil = s // 2
  
  if r == 2:
    raise Exception("Deprecated...")
    res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r].set(1)
    res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r + 2].set(1)
  elif r == 4:
    if BC == "periodic":
      for i in range(N2):
        start = i * r
        for j in range(s):
          idx = (start + j) % N1  # wrap around domain
          res_op = res_op.at[i, idx].set(1)
      res_op /= s / 4  # normalize to preserve energy scale
    else:  # Dirichlet-Neumann
      for i in range(N2):
        res_op = res_op.at[i, i * r:i * r + s].set(1)
      res_op /= s / 4
  elif r == 8:
    # stencil = 12
    for i in range(N2):
      res_op = res_op.at[i, i * r + 3:i * r + 12].set(1)
    res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r + 7].set(0)
  
  return res_op

'''
def _create_gaussian_filter(N1, N2, r, BC):
  """Create Gaussian filter operator using same edge logic as box filter"""
  res_op = jnp.zeros((N2, N1))

  if r == 2:
    raise Exception("Deprecated...")

  elif r == 4:

    sigma = r // 2 
    stencil_size = (6 * sigma) + 1
    half_width = stencil_size // 2

    for i in range(N2):
        center = i * r
        start = center - half_width
        end = center + half_width + 1

        weight_accum = {}  # dictionary to accumulate weights at each fine index

        for j in range(start, end):
            if BC == "periodic":
                j_wrapped = j % N1
            elif 0 <= j < N1:
                j_wrapped = j
            else:
                continue  # skip out-of-bounds for Dirichlet-Neumann

            distance = j - center  # always use unwrapped distance
            weight = jnp.exp(-0.5 * (distance / sigma) ** 2)

            # Accumulate (sum) weights in case of repeated indices
            if j_wrapped in weight_accum:
                weight_accum[j_wrapped] += weight
            else:
                weight_accum[j_wrapped] = weight

        # Normalize the row
        total_weight = sum(weight_accum.values()) + 1e-12
        for j_wrapped, weight in weight_accum.items():
            res_op = res_op.at[i, j_wrapped].set(weight / total_weight)

  elif r == 8:
    raise Exception("Deprecated...")

  return res_op
'''

def _create_gaussian_filter(N1, N2, r, BC):
  """Create Gaussian filter operator, handling periodic and non-periodic BCs correctly."""
  res_op = jnp.zeros((N2, N1))

  sigma = r / 2
  truncate = 3.0
  stencil_size = int(2 * truncate * sigma + 1)

  if BC == "periodic":
    # Manual stencil logic (as before)
    half_width = stencil_size // 2
    for i in range(N2):
      center = i * r
      start = center - half_width
      end = center + half_width + 1

      weight_accum = {}

      for j in range(start, end):
        j_wrapped = j % N1
        distance = j - center
        weight = jnp.exp(-0.5 * (distance / sigma) ** 2)

        if j_wrapped in weight_accum:
          weight_accum[j_wrapped] += weight
        else:
          weight_accum[j_wrapped] = weight

        total_weight = sum(weight_accum.values()) + 1e-12
        for j_wrapped, weight in weight_accum.items():
            res_op = res_op.at[i, j_wrapped].set(weight / total_weight)

  elif BC == "Dirichlet-Neumann":
      # Use scipy.ndimage for filtering with reflect BC
      identity = np.eye(N1)
      filtered_rows = gaussian_filter1d(identity, sigma=sigma, axis=1, mode='reflect', truncate=truncate)
      
      for i in range(N2):
          res_op = res_op.at[i].set(filtered_rows[i * r])

  else:
      raise ValueError(f"Unsupported boundary condition: {BC}")

  return res_op

def _create_spectral_filter(N1, N2, r, BC):
  """Create spectral cutoff filter operator"""
  if BC == "Dirichlet-Neumann":
    # For non-periodic BC, use DCT-based spectral filter
    # This is more complex - for now, fall back to a smooth approximation
    res_op = _create_smooth_spectral_filter_nonperiodic(N1, N2, r)
  else:
    # For periodic BC, use FFT-based spectral filter
    res_op = _create_spectral_filter_periodic(N1, N2, r)
  
  return res_op

'''
def _create_spectral_filter_periodic(N1, N2, r):
  """FFT-based spectral filter for periodic BCs (sharp cutoff in frequency space)"""
  # For spectral filtering, we want to:
  # 1. FFT the fine grid data
  # 2. Keep only the low frequencies that fit on coarse grid  
  # 3. IFFT and subsample
  
  # The cutoff should be based on the coarse grid Nyquist frequency
  cutoff = N2 // 2
  
  # Create the spectral filter as a matrix operation
  res_op = jnp.zeros((N2, N1))
  
  for i in range(N2):
    # Create a unit impulse at the i-th coarse grid point
    impulse = jnp.zeros(N1)
    impulse = impulse.at[i * r].set(1.0)
    
    # Apply spectral filtering:
    # 1. FFT of the impulse
    impulse_hat = jnp.fft.fft(impulse)
    
    # 2. Create frequency mask - keep low frequencies only
    freqs = jnp.fft.fftfreq(N1, 1.0) * N1  # frequency indices
    mask = jnp.abs(freqs) <= cutoff
    
    # 3. Apply filter and transform back
    filtered_hat = impulse_hat * mask
    filtered = jnp.fft.ifft(filtered_hat).real
    
    # 4. The filtered impulse gives us the i-th row of the restriction operator
    res_op = res_op.at[i].set(filtered[::r])  # subsample every r points
  
  return res_op
'''
'''
def _create_spectral_filter_periodic(N1, N2, r):
  """
  Create a spectral restriction operator res_op ∈ ℝ^{N2×N1}
  by filtering identity rows (no impulses!) with a sharp low-pass mask.

  Assumes periodic boundary conditions.
  """
  # Identity matrix of shape (N1, N1)
  I = jnp.eye(N1)

  # FFT of identity matrix (each row is a unit input)
  I_hat = jnp.fft.fft(I, axis=1)

  # Define cutoff frequency
  cutoff = N2 // 2
  freqs = jnp.fft.fftfreq(N1) * N1
  mask = jnp.where(jnp.abs(freqs) <= cutoff, 1.0, 0.0)

  # Apply spectral mask across all rows
  I_hat_filtered = I_hat * mask  # broadcasted along rows

  # Inverse FFT to get filtered signals (still (N1, N1))
  filtered = jnp.fft.ifft(I_hat_filtered, axis=1).real

  # Subsample every r-th row to form restriction operator
  res_op = filtered[::r, :]

  # Normalize each row to sum to 1 (mass-preserving)
  res_op /= jnp.sum(res_op, axis=1, keepdims=True)

  return res_op
'''
'''
def _create_spectral_filter_periodic(N1, N2, r):
#Doesn't work  
    """
    Create a spectral restriction operator R ∈ ℝ^{N2×N1}
    using explicit RFFT/IRFFT matrices and a sharp low-pass mask.
    Periodic BCs.
    """
    # Step 1: Fine-grid RFFT matrix (freq x space)
    fft_mat_fine = jnp.fft.rfft(jnp.eye(N1)).T   # shape: (N1//2+1, N1)

    # Step 2: Coarse-grid IRFFT matrix (space x freq)
    fft_mat_coarse = jnp.fft.irfft(jnp.eye(N2//2 + 1), n=N2).T  # shape: (N2, N2//2+1)

    # Step 3: Spectral mask S (coarse freq size × fine freq size). Cutoff freq = N2//2
    S = jnp.zeros((N2//2 + 1, N1//2 + 1))
    S = S.at[:N2//2 + 1, :N2//2 + 1].set(jnp.eye(N2//2 + 1))

    # Step 4: Compose restriction operator: R = F2^-1 @ S @ F1
    res_op = (fft_mat_coarse @ S @ fft_mat_fine).real

    # Step 5: Normalize rows to preserve mass
    res_op /= res_op.sum(axis=1, keepdims=True)

    return res_op

'''

def _create_spectral_filter_periodic(N1, N2, r):
    """
    Create a spectral restriction operator res_op ∈ ℝ^{N2×N1}
    using SciPy's Fourier-method resampling (periodic BCs).

    This internally does: FFT -> truncate high freqs -> IFFT -> length N2.

    Parameters
    ----------
    N1 : int
        Fine grid size.
    N2 : int
        Coarse grid size.
    r  : int
        Coarsening ratio (N1 / N2).

    Returns
    -------
    res_op : (N2, N1) ndarray
        Spectral restriction operator. Multiplying by a fine-grid vector
        gives the coarse-grid result.
    """
    # Identity matrix on fine grid
    I = np.eye(N1, dtype=float)  # shape: (N1, N1)

    # Resample each fine-grid basis vector down to N2 via Fourier method
    # axis=1 means we're changing the length of the signal dimension
    filtered = resample(I, N2, axis=1)

    # Transpose so shape is (N2, N1) rather than (N1, N2)
    res_op = filtered.T

    # Normalize each row so a constant field stays constant
    res_op /= res_op.sum(axis=1, keepdims=True)

    return res_op


def _create_smooth_spectral_filter_nonperiodic(N1, N2, r):
  """Smooth spectral-like filter for non-periodic BC using same structure as Gaussian filter"""
  res_op = jnp.zeros((N2, N1))
  
  if r == 2:
    raise Exception("Deprecated...")

  elif r == 4:
    stencil_size = 2 * r + 1  # 9
    half_width = stencil_size // 2  # 4

    for i in range(N2):
      center = i * r
      start = center - half_width
      end = center + half_width + 1

      weight_accum = {}  # dictionary to accumulate weights at each fine index

      for j in range(start, end):
        if 0 <= j < N1:
          j_wrapped = j
        else:
          continue  # skip out-of-bounds for Dirichlet-Neumann

        distance = j - center  # always use unwrapped distance
        # Use sinc approximation
        weight = 0.5 * (1 + jnp.cos(jnp.pi * distance / r))

        # Accumulate (sum) weights in case of repeated indices
        if j_wrapped in weight_accum:
          weight_accum[j_wrapped] += weight
        else:
          weight_accum[j_wrapped] = weight

      # Normalize the row
      total_weight = sum(weight_accum.values()) + 1e-12
      for j_wrapped, weight in weight_accum.items():
          res_op = res_op.at[i, j_wrapped].set(weight / total_weight)

  elif r == 8:
    raise Exception("Deprecated...")

  return res_op


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
    
    if filter_type == "box":
      stencil_size = config.sim.get('stencil_size', 7)  # default to 7 for backward compatibility
      if stencil_size % 2 == 0:
        raise ValueError("Stencil size must be odd")
      # Updated box filter implementation with stencil size parameter
      res_op = _create_box_filter(N1, N2, r, BC, stencil_size)
      res_op /= r
    elif filter_type == "gaussian":
      res_op = _create_gaussian_filter(N1, N2, r, BC)
    elif filter_type == "spectral":
      res_op = _create_spectral_filter(N1, N2, r, BC)
    else:
      raise ValueError(f"Unknown filter_type: {filter_type}")
    
    int_op = jnp.zeros((N1, N2))
    int_op = jnp.linalg.pinv(res_op)
    print('res_op: ', res_op)
    print('int_op: ', int_op)
    assert jnp.allclose(res_op @ int_op, jnp.eye(N2))  #atol=1e-3
    assert jnp.allclose(res_op.sum(axis=-1), jnp.ones(N2)) #atol=1e-6

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
