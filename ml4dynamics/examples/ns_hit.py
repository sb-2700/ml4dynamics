"""
simulation example for 2D incompressible Navier-Stokes equations
using spectral method

For both fine and coarse simulation to be stable, use
Re = 2000
r = 2 (N_fine = 256, N_coarse = 128)
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import yaml
from box import Box

from ml4dynamics.utils import calc_utils, utils, viz_utils

with open(f"config/ns_hit.yaml", "r") as file:
  config_dict = yaml.safe_load(file)
config = Box(config_dict)
Re = config.sim.Re
n = config.sim.n
T = config.sim.T
dt = config.sim.dt
L = config.sim.L
model_fine, model_coarse = utils.create_fine_coarse_simulator(config)
case_num = config.sim.case_num
writeInterval = 1

model_fine.w_hat = utils.hit_init_cond("spec_random", model_fine)
w_hat = jnp.roll(model_fine.w_hat, shift=model_fine.N // 2,
                 axis=0)[model_fine.N // 2 -
                         model_coarse.N // 2:model_fine.N // 2 +
                         model_coarse.N // 2, :model_coarse.N // 2 + 1]
w_hat = jnp.roll(w_hat, shift=-model_coarse.N // 2, axis=0)
model_fine.set_x_hist(model_fine.w_hat, model_fine.CN)
model_coarse.set_x_hist(w_hat, model_coarse.CN)

n_plot = 3
plot_interval = model_fine.step_num // n_plot**2
im_array1 = np.zeros((n_plot, n_plot, n, n))
im_array2 = np.zeros((n_plot, n_plot, model_coarse.N, model_coarse.N))
title_array = []
for i in range(n_plot**2):
  im_array1[i // 3, i % 3] = model_fine.x_hist[i * plot_interval]
  im_array2[i // 3, i % 3] = model_coarse.x_hist[i * plot_interval]
  title_array.append(f"t={i * plot_interval * dt:.2f}")

utils.plot_with_horizontal_colorbar(
  im_array1, (12, 12), title_array, "results/fig/ns_hit_fine.png"
)
utils.plot_with_horizontal_colorbar(
  im_array2, (12, 12), title_array, "results/fig/ns_hit_coarse.png"
)

interval = 1000
tau = calc_utils.calc_reynolds_stress(
  jnp.array(model_fine.u_hist[..., None, :]), interval
)
im_array = np.zeros((2, 2, n, n))
title_array = [r"$\tau_{xx}$", r"$\tau_{xy}$", r"$\tau_{yx}$", r"$\tau_{yy}$"]
for i in range(2):
  for j in range(2):
    im_array[i, j] = tau[:, :, 0, i, j]
utils.plot_with_horizontal_colorbar(
  im_array, (12, 12), title_array, "results/fig/ns_hit_reynolds.png"
)
viz_utils.plot_psd_cmp(
  model_fine.u_hist, model_coarse.u_hist, [model_fine.dx, model_fine.dx],
  [model_coarse.dx, model_coarse.dx], ""
)
breakpoint()
