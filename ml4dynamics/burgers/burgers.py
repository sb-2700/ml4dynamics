import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=15)
jax.config.update("jax_enable_x64", True)


def main():
  L = 2 * jnp.pi
  nx = 256
  dx = L / nx
  dt = 0.001
  T = 1
  step_num = int(T / dt)
  x = jnp.linspace(0, L, nx, endpoint=False)
  u0 = jnp.sin(x)

  def burgers_godunov(u0):
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

    u_godunov = np.zeros((step_num, nx))
    for i in range(step_num):
        u_godunov[i] = u0
        u0 = godunov_step(u0)

    return u_godunov
  
  u_godunov = burgers_godunov(u0)

  def burgers_spectral():
    
    dealias = True
    if dealias:
      k_max = nx//2 * 2//3

    k = jnp.fft.fftfreq(nx, d=1.0/nx) * 2*jnp.pi/L
    k = jnp.fft.fftshift(k)

    @jax.jit
    def spectral_step(u, dt):
      u_hat = jnp.fft.fft(u)
      
      def rhs(u_hat):
        if dealias:
          u_hat_trunc = jnp.fft.fftshift(u_hat)
          u_hat_trunc = jnp.where(jnp.abs(k) > k_max, 0, u_hat_trunc)
          u_hat_trunc = jnp.fft.ifftshift(u_hat_trunc)
          u = jnp.fft.ifft(u_hat_trunc)
        nonlinear = 0.5 * u**2
        nonlinear_hat = jnp.fft.fft(nonlinear)
        return -1j * k * nonlinear_hat
      
      k1 = rhs(u_hat)
      k2 = rhs(u_hat + 0.5*dt*k1)
      k3 = rhs(u_hat + 0.5*dt*k2)
      k4 = rhs(u_hat + dt*k3)
      
      u_hat_new = u_hat + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
      return u_hat_new

    u_spectral = np.zeros((step_num, nx))
    for i in range(step_num):
        u_spectral[i] = u0
        u0 = spectral_step(u0)

    return u_spectral


  u_spectral = burgers_spectral(u0, step_num, dt)
  breakpoint()

  mass_initial = jnp.sum(u0) * dx
  mass_final = jnp.sum(u_godunov[-1]) * dx
  print(f"mass conservation: {abs(mass_final - mass_initial):.2e}")

  plt.figure(figsize=(10,6))
  plt.plot(x, u_godunov[-1], label='Godunov')
  # plt.plot(x, u_spectral, '--', label='Spectral')
  plt.title('Burgers Equation Solutions')
  plt.xlabel('x')
  plt.ylabel('u')
  plt.legend()
  plt.savefig('results/fig/burgers_solution.pdf')

  # def track_conservation(method, u0, nt):
  #     mass = []
  #     u = u0.copy() if method == 'godunov' else jnp.fft.fft(u0)
  #     for _ in range(nt):
  #         if method == 'godunov':
  #             u = godunov_step(u, dt, dx)
  #             mass.append(jnp.sum(u)*dx)
  #         # else:
  #         #     u = spectral_step(u, dt)
  #         #     mass.append(jnp.sum(jnp.fft.ifft(u).real)*dx)
  #     return jnp.array(mass)


if __name__ == "__main__":
  main()
