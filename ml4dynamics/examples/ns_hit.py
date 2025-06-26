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
from jax import random
from jax.scipy.linalg import svd
from matplotlib import pyplot as plt

from ml4dynamics.dataset_utils import dataset_utils
from ml4dynamics.utils import calc_utils, utils, viz_utils

with open(f"config/ns_hit.yaml", "r") as file:
  config_dict = yaml.safe_load(file)
config = Box(config_dict)
Re = config.sim.Re
n = config.sim.n
T = config.sim.T
dt = config.sim.dt
L = config.sim.L
rng = random.PRNGKey(config.sim.seed)
model_fine, model_coarse = utils.create_fine_coarse_simulator(config)
case_num = config.sim.case_num
writeInterval = 1

model_fine.w_hat = utils.hit_init_cond("spec_random", model_fine, rng)
w_hat = jnp.roll(model_fine.w_hat, shift=model_fine.N // 2,
                 axis=0)[model_fine.N // 2 -
                         model_coarse.N // 2:model_fine.N // 2 +
                         model_coarse.N // 2, :model_coarse.N // 2 + 1]
w_hat = jnp.roll(w_hat, shift=-model_coarse.N // 2, axis=0)
model_fine.set_x_hist(model_fine.w_hat, model_fine.CN)
model_coarse.set_x_hist(w_hat, model_coarse.CN)
x_hist = model_coarse.x_hist
res_fn, _ = dataset_utils.res_int_fn(config_dict)
model_coarse.set_x_hist(
  jnp.fft.rfft2(res_fn(jnp.fft.irfft2(model_fine.w_hat)[..., None])[..., 0]),
  model_coarse.CN
)

n_plot = 8
plot_interval = model_fine.step_num // n_plot
im_list1 = []
im_list2 = []
im_list3 = []
title_array = []
for i in range(n_plot):
  im_list1.append(model_fine.x_hist[i * plot_interval])
  im_list2.append(x_hist[i * plot_interval])
  im_list3.append(model_coarse.x_hist[i * plot_interval])
  title_array.append(f"t={i * plot_interval * dt:.2f}")

utils.plot_with_horizontal_colorbar(
  im_list1 + im_list2 + im_list3, title_array + title_array + title_array,
  [3, n_plot], None, "results/fig/ns_hit.png", 100
)
inputs = model_fine.x_hist.reshape(model_fine.x_hist.shape[0], -1)
_, s, vt = svd(
  jnp.array(inputs - jnp.mean(inputs, axis=0)[None]), full_matrices=False
)
print(s)
"""NOTE: the singular values decay very slowly for KS equation"""
t_array = np.linspace(0, T, model_coarse.step_num)
filtered_x = jax.vmap(res_fn)(model_fine.x_hist[..., None])
viz_utils.plot_corr_over_t(
  [filtered_x[..., 0], model_coarse.x_hist], [''], t_array, "ns_hit"
)

breakpoint()
s = np.array(s)
vt = np.array(vt)
_ = plt.figure(figsize=(12, 6))
transform_x = inputs @ vt.T
plt.subplot(121)
plt.scatter(transform_x[:, 0], transform_x[:, 1], s=0.1)
plt.subplot(122)
plt.plot(s[:100])
plt.yscale("log")
plt.savefig(f"results/fig/pod_hit.png")
plt.close()

interval = 1000
tau = calc_utils.calc_reynolds_stress(
  jnp.array(model_fine.u_hist[..., None, :]), interval
)
im_array = np.zeros((2, 2, n_plot, n_plot))
title_array = [r"$\tau_{xx}$", r"$\tau_{xy}$", r"$\tau_{yx}$", r"$\tau_{yy}$"]
for i in range(2):
  for j in range(2):
    im_array[i, j] = tau[:, :, 0, i, j]
im_list = []
for i in range(2):
  for j in range(2):
    im_list.append(tau[:, :, 0, i, j])
utils.plot_with_horizontal_colorbar(
  im_list, title_array, [2, 2], None, "results/fig/ns_hit_reynolds.png", 100
)
viz_utils.plot_psd_cmp(
  model_fine.u_hist, model_coarse.u_hist, [model_fine.dx, model_fine.dx],
  [model_coarse.dx, model_coarse.dx], ""
)
breakpoint()
