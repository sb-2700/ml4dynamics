"""
Test the data generator, make sure the data generated is physcial and
correct. More rigorous tests should be done for mesh-refinement study.
TODO: functional programming so we can call the function we want
"""
import jax
import jax.numpy as jnp
import pytest
from jax import random

from ml4dynamics import utils
from ml4dynamics.dynamics import KS


jax.config.update('jax_enable_x64', True)

# @pytest.mark.parametrize(
#     ("hw", "param_count"),
#     [
#         ((128, 128), 34_491_599),
#         # It's fully convolutional => same parameter number.
#         ((256, 256), 34_491_599),
#     ],
#   )

# def test_turing_pattern():
#     """Test whether the set of parameter will generate Turing pattern for RD equation
#     TODO
#     """
#     assert True


@pytest.mark.skip
def test_reaction_diffusion_equation_solver():
  r"""We check the numerical solution with the following analytic solution
    test case for RD equation:

    $$
        u(x, y, t) = sin(2\pi x/L)
        v(x, y, t) = sin(2\pi y/L)
        D = [[1, 0], [0, 1]]
        alpha = 0.01
        beta = 1.0
        s_u(x, y, t) = ((2\pi/L)^2-1)*u + v**3 + v - 0.01
        s_v(x, y, t) = (2\pi/L)^2*v + v - u
    $$

    NOTE: discretization is chosen so that all the simulations do not blow-up
    """

  n = 32
  widthx = 6.4
  step_num = 1000
  warm_up = 0
  writeInterval = 1
  dx = widthx / n
  alpha = 0.01
  beta = 1.0
  gamma = 1.0
  dt = 0.01
  omega = 2 * jnp.pi / widthx

  utils.assembly_RDmatrix(n, dt, dx, beta=beta, gamma=gamma, d=1.0)c
  mesh_1d = jnp.linspace(0, widthx, n + 1)
  u = jnp.sin(mesh_1d[:-1] * omega).reshape(n, 1) + jnp.zeros((1, n))
  v = jnp.sin(mesh_1d[:-1] * omega).reshape(1, n) + jnp.zeros((n, 1))
  source = jnp.zeros((2, n, n))
  source = source.at[0].set((omega**2 - 1) * u + u**3 + v - alpha)
  source = source.at[1].set(omega**2 * v + v - u)

  u_hist, v_hist, _ = utils.RD_adi(
    u,
    v,
    dt,
    source=source,
    alpha=alpha,
    beta=beta,
    step_num=step_num + warm_up,
    writeInterval=writeInterval
  )
  assert jnp.allclose(u_hist[-1], u_hist[0], atol=2e-3) and jnp.allclose(
    v_hist[-1], v_hist[0], atol=2e-3
  )

  u_hist, v_hist = utils.RD_exp(
    u,
    v,
    dt,
    source=source,
    alpha=alpha,
    beta=beta,
    step_num=step_num + warm_up,
    writeInterval=writeInterval
  )
  assert jnp.allclose(u_hist[-1], u_hist[0], atol=2e-3) and jnp.allclose(
    v_hist[-1], v_hist[0], atol=2e-3
  )

  u_hist, v_hist = utils.RD_semi(
    u,
    v,
    dt,
    source=source,
    alpha=alpha,
    beta=beta,
    step_num=step_num + warm_up,
    writeInterval=writeInterval
  )
  assert jnp.allclose(u_hist[-1], u_hist[0], atol=2e-3) and jnp.allclose(
    v_hist[-1], v_hist[0], atol=2e-3
  )

# @pytest.mark.skip
def test_kuramoto_sivashinsky_equation_solver():
  r"""
  periodic BC:
  u(x, t) = \sin(x)
  s(x, t) = -0.5 \sin(2x) - 0.8 \cos(x)

  Dirichlet-Neumann BC:
  u(x, t) = 2\sin(x) - \sin(2x)
  s(x, t) = 10\sin(2x) - 1.6\cos(x) - \sin(4x) + 1.6\cos(2x) + 3\sin(3x) - \sin(x)
  """

  # periodic BC
  nu = 1
  c = 1.0
  L = 2 * jnp.pi
  T = 1
  init_scale = 1.0
  N = 1024
  dt = 0.005
  step_num = int(T / dt) 
  key = random.PRNGKey(0)
  ks = KS(
    N=N,
    T=T,
    dt=dt,
    init_scale=init_scale,
    L=L,
    nu=nu,
    c=c,
    rng=key,
  )
  ks.source = -c * jnp.cos(jnp.linspace(0, L - L/N, N)) -\
    0.5 * jnp.sin(2 * jnp.linspace(0, L - L/N, N))
  x = jnp.sin(jnp.linspace(0, L - L/N, N))
  start = jnp.sin(jnp.linspace(0, L - L/N, N))
  for _ in range(step_num):
    x = ks.CN_FEM_test(x)
  assert jnp.allclose(x, start, atol=1e-5)

  # Dirichlet-Neumann BC
  N = 1024
  dt = 0.005
  step_num = int(T / dt) 
  ks = KS(
    N=N-1,
    T=T,
    dt=dt,
    init_scale=init_scale,
    L=L,
    nu=nu,
    c=c,
    BC="Dirichlet-Neumann",
    rng=key,
  )
  ks.source = -2 * c * jnp.cos(jnp.linspace(L/N, L - L/N, N-1)) +\
    10 * jnp.sin(2 * jnp.linspace(L/N, L - L/N, N-1)) +\
    2 * c * jnp.cos(2 * jnp.linspace(L/N, L - L/N, N-1)) -\
    jnp.sin(4 * jnp.linspace(L/N, L - L/N, N-1)) +\
    3 * jnp.sin(3 * jnp.linspace(L/N, L - L/N, N-1)) -\
    jnp.sin(jnp.linspace(L/N, L - L/N, N-1))
  x = 2 * jnp.sin(jnp.linspace(L/N, L - L/N, N-1)) -\
    jnp.sin(2 * jnp.linspace(L/N, L - L/N, N-1))
  start = 2 * jnp.sin(jnp.linspace(L/N, L - L/N, N-1)) -\
    jnp.sin(2 * jnp.linspace(L/N, L - L/N, N-1))
  for _ in range(step_num):
    x = ks.CN_FEM_test(x)
  assert jnp.allclose(x, start, atol=5e-5)
