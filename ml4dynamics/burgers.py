import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def main():
  L = 2 * jnp.pi
  nx = 256
  dx = L / nx
  dt = 0.001
  T = 1
  step_num = int(T / dt)
  x = jnp.linspace(0, L, nx, endpoint=False)
  u0 = jnp.sin(x)

  @jax.jit
  def godunov_step(u: jnp.ndarray) -> jnp.ndarray:
    """Godunov's scheme"""
    
    # Godunov通量计算 (Riemann解)
    # 对于Burgers方程，Godunov通量为 min(f(u_l), f(u_r)) if u_l > u_r else max(...)
    u_left = jnp.roll(u, 1)
    u_right = u
    flux = jnp.where(u_left > u_right, 
                    jnp.minimum(0.5*u_left**2, 0.5*u_right**2),
                    jnp.maximum(0.5*u_left**2, 0.5*u_right**2))
    
    u_new = u - dt/dx * (flux - jnp.roll(flux, 1))
    return u_new

  u_godunov = np.zeros((step_num, nx))
  for i in range(step_num):
      u_godunov[i] = u0
      u0 = godunov_step(u0)

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

  # mass_godunov = track_conservation('godunov', u0, step_num)
  # # mass_spectral = track_conservation('spectral', u0, step_num)

  # plt.figure()
  # plt.semilogy(jnp.abs(mass_godunov - mass_initial), label='Godunov')
  # # plt.semilogy(jnp.abs(mass_spectral - mass_initial), label='Spectral')
  # plt.title('Mass Conservation Error')
  # plt.xlabel('Time Step')
  # plt.ylabel('|Mass Error|')
  # plt.legend()
  # plt.show()

if __name__ == "__main__":
  main()
