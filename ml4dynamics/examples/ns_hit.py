import jax.numpy as jnp
import numpy as np
import yaml
from box import Box
from jax import random

from ml4dynamics.utils import plot_with_horizontal_colorbar
from ml4dynamics.dynamics import ns_hit
from pyfoam.utils import calc_utils

with open(f"config/ns_hit.yaml", "r") as file:
  config_dict = yaml.safe_load(file)
config = Box(config_dict)
Re = config.sim.Re
n = config.sim.n
T = config.sim.T
dt = config.sim.dt
L = config.sim.L
model = ns_hit(L=L * jnp.pi, N=n, nu=1/Re, T=T, dt=dt, init_scale=n**(1.5))
case_num = config.sim.case_num
writeInterval = 1

model.w_hat = jnp.zeros((model.N, model.N//2+1))
f0 = int(jnp.sqrt(n/2)) # init frequency
model.w_hat = model.w_hat.at[:f0, :f0].set(
  random.normal(random.PRNGKey(0), (f0, f0)) * model.init_scale
)
model.set_x_hist(model.w_hat, model.CN)

n_plot = 3
plot_interval = model.step_num // n_plot**2
im_array = np.zeros((n_plot, n_plot, n, n))
title_array = []
for i in range(n_plot**2):
  im_array[i//3, i%3] = model.x_hist[i * plot_interval]
  title_array.append(f"t={i * plot_interval * dt:.2f}")

plot_with_horizontal_colorbar(
  im_array, (12, 12), title_array, "ns_hit.png"
)

tau = calc_utils.calc_reynolds_stress(model.u_hist[..., None, :], 900)
im_array = np.zeros((2, 2, n, n))
title_array = [r"$\tau_{xx}$", r"$\tau_{xy}$", r"$\tau_{yx}$", r"$\tau_{yy}$"]
for i in range(2):
  for j in range(2):
    im_array[i, j] = tau[:, :, 0, i, j]
plot_with_horizontal_colorbar(
  im_array, (12, 12), title_array, "ns_hit_reynolds.png"
)

y_grid = jnp.linspace(0, L * jnp.pi, n, endpoint=False)
breakpoint()
calc_utils.power_spec(
  model.u_hist[..., None, :], y_grid, model.dx, 1, (1, ), "psd.png"
)

breakpoint()
