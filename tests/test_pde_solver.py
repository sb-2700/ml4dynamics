"""
Test the data generator, make sure the data generated is physcial and
correct. May already implement the mesh refinement study in our py
notebook
"""

# We need to use functional programming so we can call the function we want
import jax
import jax.numpy as jnp
import pytest
from jax import random
from matplotlib import pyplot as plt

from src import utils

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


def test_reaction_diffusion_equation_solver():
  """We check the numerical solution with the following analytic solution
    test case for RD equation:
        u(x, y, t) = sin(2\pi x/L)
        v(x, y, t) = sin(2\pi y/L)
        D = [[1, 0], [0, 1]]
        alpha = 0.01
        beta = 1.0
        s_u(x, y, t) = ((2\pi/L)^2-1)*u + v**3 + v - 0.01
        s_v(x, y, t) = (2\pi/L)^2*v + v - u
    """

  n = 32
  widthx = 6.4
  step_num = 100
  warm_up = 0
  writeInterval = 1
  dx = widthx / n
  alpha = 0.01
  beta = 1.0
  gamma = 1.0
  dt = 0.01
  tol = 2e-6
  omega = 2 * jnp.pi / widthx

  utils.assembly_RDmatrix(n, dt, dx, beta=beta, gamma=gamma, d=1.0)

  mesh_1d = jnp.linspace(0, widthx, n + 1)
  u = jnp.sin(mesh_1d[:-1] * omega).reshape(n, 1) + jnp.zeros((1, n))
  v = jnp.sin(mesh_1d[:-1] * omega).reshape(1, n) + jnp.zeros((n, 1))
  source = jnp.zeros((2, n, n))
  source = source.at[0].set((omega**2 - 1) * u + u**3 + v - alpha)
  source = source.at[1].set(omega**2 * v + v - u)

  # test the alternating direction implicit (ADI) method
  u_hist, v_hist = utils.RD_adi(
    u,
    v,
    dt,
    source=source,
    alpha=alpha,
    beta=beta,
    step_num=step_num + warm_up,
    writeInterval=writeInterval
  )
  assert jnp.mean(
    (u_hist[-1] - u_hist[0])**2 + (v_hist[-1] - v_hist[0])**2
  ) < tol

  # test the explicit method
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

  assert jnp.mean(
    (u_hist[-1] - u_hist[0])**2 + (v_hist[-1] - v_hist[0])**2
  ) < tol

  # test the semi-implicit method
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

  assert jnp.mean(
    (u_hist[-1] - u_hist[0])**2 + (v_hist[-1] - v_hist[0])**2
  ) < tol


def test_navier_stokes_equation_solver():
  """We check the numerical solution with the following analytic solution
    test case for NS equation:
        u(x, y, t) = sin(2\pi x/L)
        v(x, y, t) = sin(2\pi y/L)
        D = [[1, 0], [0, 1]]
        alpha = 0.01
        beta = 1.0
        s_u(x, y, t) = ((2\pi/L)^2-1)*u + v**3 + v - 0.01
        s_v(x, y, t) = (2\pi/L)^2*v + v - u
    """

  nx = 128
  ny = 32
  widthx = 4
  widthy = 1
  step_num = 100
  warm_up = 0
  writeInterval = 1
  dx = widthx / nx
  dy = widthy / ny
  dt = 0.01
  tol = 2e-6
  omega = 2 * jnp.pi / widthx

  utils.assembly_NSmatrix(nx, ny, dt, dx, dy)

  mesh_1d = jnp.linspace(0, widthx, n + 1)
  u = jnp.sin(mesh_1d[:-1] * omega).reshape(n, 1) + jnp.zeros((1, n))
  v = jnp.sin(mesh_1d[:-1] * omega).reshape(1, n) + jnp.zeros((n, 1))
  source = jnp.zeros((2, n, n))
  source = source.at[0].set((omega**2 - 1) * u + u**3 + v - alpha)
  source = source.at[1].set(omega**2 * v + v - u)

  # test the alternating direction implicit (ADI) method
  u_hist, v_hist = utils.projection_correction(
    u,
    v,
    dt,
    source=source,
    alpha=alpha,
    beta=beta,
    step_num=step_num + warm_up,
    writeInterval=writeInterval
  )
  assert jnp.mean(
    (u_hist[-1] - u_hist[0])**2 + (v_hist[-1] - v_hist[0])**2
  ) < tol
