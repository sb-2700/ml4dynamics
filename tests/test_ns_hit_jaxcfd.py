import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import xarray

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral

import dataclasses

# physical parameters
viscosity = 1e-3
max_velocity = 7
n1 = 256
r = 4
n2 = n1 // r

# fine grid simulation
grid_f = grids.Grid((n1, n1), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
dt_f = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid_f)

# setup step function using crank-nicolson runge-kutta order 4
smooth = True # use anti-aliasing 
step_fn = spectral.time_stepping.crank_nicolson_rk4(
    spectral.equations.NavierStokes2D(viscosity, grid_f, smooth=smooth), dt_f)

# run the simulation up until time 25.0 but only save 10 frames for visualization
final_time = 25.0
outer_steps = 10
inner_steps = (final_time // dt_f) // 10

trajectory_fn = cfd.funcutils.trajectory(
    cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

# create an initial velocity field and compute the fft of the vorticity.
# the spectral code assumes an fft'd vorticity for an initial state
v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(42), grid_f, max_velocity, 4)
vorticity0 = cfd.finite_differences.curl_2d(v0).data
vorticity_hat0 = jnp.fft.rfftn(vorticity0)

_, trajectory = trajectory_fn(vorticity_hat0)

spatial_coord = jnp.arange(grid_f.shape[0]) * 2 * jnp.pi / grid_f.shape[0] # same for x and y
coords = {
  'time': dt_f * jnp.arange(outer_steps) * inner_steps,
  'x': spatial_coord,
  'y': spatial_coord,
}
xarray.DataArray(
    jnp.fft.irfftn(trajectory, axes=(1,2)), 
    dims=["time", "x", "y"], coords=coords
).plot.imshow(
  col='time', col_wrap=5, cmap=sns.cm.icefire, robust=True
)
plt.savefig('ns_hit_jaxcfd_fine.png')
plt.close()

grid_c = grids.Grid((n2, n2), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
dt_c = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid_c)

# setup step function using crank-nicolson runge-kutta order 4
smooth = True # use anti-aliasing 
step_fn = spectral.time_stepping.crank_nicolson_rk4(
    spectral.equations.NavierStokes2D(viscosity, grid_c, smooth=smooth), dt_c)

# run the simulation up until time 25.0 but only save 10 frames for visualization
inner_steps = (final_time // dt_c) // 10

trajectory_fn = cfd.funcutils.trajectory(
    cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

result = jnp.zeros((n2, n2))
for k in range(r):
  for j in range(r):
    result += vorticity0[k::r, j::r] / (r**2)
plt.subplot(121)
plt.imshow(vorticity0, cmap=sns.cm.icefire)
plt.subplot(122)
plt.imshow(result, cmap=sns.cm.icefire)
plt.savefig('ns_hit_jaxcfd_coarse_vorticity.png')
plt.close()
breakpoint()
vorticity_hat0 = jnp.fft.rfftn(result)
_, trajectory = trajectory_fn(vorticity_hat0)

spatial_coord = jnp.arange(grid_c.shape[0]) * 2 * jnp.pi / grid_c.shape[0] # same for x and y
coords = {
  'time': dt_c * jnp.arange(outer_steps) * inner_steps,
  'x': spatial_coord,
  'y': spatial_coord,
}
xarray.DataArray(
    jnp.fft.irfftn(trajectory, axes=(1,2)), 
    dims=["time", "x", "y"], coords=coords
).plot.imshow(
  col='time', col_wrap=5, cmap=sns.cm.icefire, robust=True
)
plt.savefig('ns_hit_jaxcfd_coarse.png')
breakpoint()

# physical parameters
viscosity = 1e-3
max_velocity = 7
grid = grids.Grid((256, 256), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
dt = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid)

# setup step function using crank-nicolson runge-kutta order 4
smooth = True # use anti-aliasing 


# **use predefined settings for Kolmogorov flow**
step_fn = spectral.time_stepping.crank_nicolson_rk4(
    spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt)


# run the simulation up until time 25.0 but only save 10 frames for visualization
final_time = 25.0
outer_steps = 10
inner_steps = (final_time // dt) // 10

trajectory_fn = cfd.funcutils.trajectory(
    cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

# create an initial velocity field and compute the fft of the vorticity.
# the spectral code assumes an fft'd vorticity for an initial state
v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(42), grid, max_velocity, 4)
vorticity0 = cfd.finite_differences.curl_2d(v0).data
vorticity_hat0 = jnp.fft.rfftn(vorticity0)

_, trajectory = trajectory_fn(vorticity_hat0)

spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0] # same for x and y
coords = {
  'time': dt * jnp.arange(outer_steps) * inner_steps,
  'x': spatial_coord,
  'y': spatial_coord,
}
xarray.DataArray(
    jnp.fft.irfftn(trajectory, axes=(1,2)), 
    dims=["time", "x", "y"], coords=coords).plot.imshow(
        col='time', col_wrap=5, cmap=sns.cm.icefire, robust=True);
plt.savefig('ns_hit_forced_jaxcfd.png')
breakpoint()