import os

import jax.numpy as jnp
import jax.random as random
import numpy as np

# define the restriction and interpolation operator
# TODO: try to change the restriction and projection operator to test the
# results, these operator should have test file
res_op = jnp.zeros((N2, N1))
int_op = jnp.zeros((N1, N2))
# NOTE: this restriction operator is useless for filter, as at returns
# vanishing SGS stress
res_op = res_op.at[jnp.arange(N2), jnp.arange(N2) * r + 1].set(1)
int_op = int_op.at[jnp.arange(N2) * r + 1, jnp.arange(N2)].set(1)
for i in range(N2):
  res_op = res_op.at[i, i * r:i * r + 6].set(1)
res_op /= 7
int_op = jnp.linalg.pinv(res_op)
assert jnp.allclose(res_op @ int_op, jnp.eye(N2))
  
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