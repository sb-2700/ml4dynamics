import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=15)
jax.config.update("jax_enable_x64", True)


def main():
  def burgers_godunov(u0: jnp.ndarray) -> jnp.ndarray:
    """Solving Burgers equ using Godunov's scheme"""
    @jax.jit
    def godunov_step(u: jnp.ndarray) -> jnp.ndarray:
      def godunov(ul, ur):
        fl = ul**2 / 2
        fr = ur**2 / 2
        # rarefaction wave
        f = jnp.where(ul < ur, jnp.minimum(fl, fr), 0)
        # shock wave
        f = jnp.where(ul > ur, jnp.maximum(fl, fr), f)
        # Minimum flux at u* 
        f = jnp.where((ul < 0) & (ur > 0), 0, f)
        return f
      
      f = godunov(jnp.roll(u, 1), u)
      u_new = u - dt/dx * (jnp.roll(f, -1) - f)
      return u_new

    nx = u0.shape[0]
    dx = L / nx
    u_godunov = np.zeros((step_num, nx))
    for i in range(step_num):
        u_godunov[i] = u0
        u0 = godunov_step(u0)
    return u_godunov

  def burgers_spectral(u0: jnp.ndarray) -> jnp.ndarray:
    """Solving Burgers equ using spectral method"""
    @jax.jit
    def spectral_step(u_hat):
      
      def rhs(u_hat):
        u_hat = jnp.hstack(
          [u_hat[:nx//2], jnp.zeros_like(u_hat), u_hat[nx//2:]]
        )
        u = jnp.fft.ifft(u_hat)
        if dealias:
          u_hat_trunc = jnp.fft.fftshift(u_hat)
          u_hat_trunc = jnp.where(jnp.abs(k) > k_max, 0, u_hat_trunc)
          u_hat_trunc = jnp.fft.ifftshift(u_hat_trunc)
          u = jnp.fft.ifft(u_hat_trunc)
        nonlinear = 0.5 * u**2
        nonlinear_hat = jnp.fft.fft(nonlinear)[:nx]
        return -1j * k * nonlinear_hat
      
      # k1 = rhs(u_hat)
      # k2 = rhs(u_hat + 0.5*dt*k1)
      # k3 = rhs(u_hat + 0.5*dt*k2)
      # k4 = rhs(u_hat + dt*k3)
      # return u_hat + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
      return u_hat + dt * rhs(u_hat)

    nx = u0.shape[0]
    dealias = False
    if dealias:
      # TODO: implement dealiasing
      k_max = nx//2 * 2//3
    k = jnp.fft.fftfreq(nx, d=1.0/nx) * 2*jnp.pi/L
    u_spectral = np.zeros((step_num, nx), dtype=np.complex128)
    u0_hat = jnp.fft.fft(u0)
    for i in range(step_num):
        u_spectral[i] = u0_hat
        u0_hat = spectral_step(u0_hat)

    return u_spectral

  def track_conservation(method, u0, nt):
    mass = []
    u = u0.copy() if method == 'godunov' else jnp.fft.fft(u0)
    for _ in range(nt):
        if method == 'godunov':
            u = godunov_step(u, dt, dx)
            mass.append(jnp.sum(u)*dx)
        else:
            u = spectral_step(u, dt)
            mass.append(jnp.sum(jnp.fft.ifft(u).real)*dx)
    return jnp.array(mass)
  
  def ic(x):
    """Initial condition"""
    return jnp.sin(x) + jnp.sin(4*x)

  L = 2 * jnp.pi
  dt = 0.001
  T = 1
  step_num = int(T / dt)

  grids = [16, 32, 64, 128]
  u_godunov = burgers_godunov(
    ic(jnp.linspace(0, L, grids[-1], endpoint=False))
  )

  plt.figure(figsize=(10,6))
  plt.plot(
    jnp.linspace(0, L, grids[-1], endpoint=False),
    u_godunov[-1],
    label='Godunov'
  )
  for grid in grids:
    x = jnp.linspace(0, L, grid, endpoint=False)
    u_spectral = burgers_spectral(ic(x))
    plt.plot(x, jnp.fft.ifft(u_spectral[-1]).real, label=f'Spectral {grid}')
    assert jnp.isclose(
      jnp.sum(ic(x)), jnp.sum(jnp.fft.ifft(u_spectral[-1]).real)
    )
  
  plt.title('Burgers Equation Solutions')
  plt.xlabel('x')
  plt.ylabel('u')
  plt.legend()
  plt.savefig('results/fig/burgers_solution.png')


if __name__ == "__main__":
  main()
