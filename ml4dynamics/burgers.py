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
  
  # u_godunov = burgers_godunov(u0)

  # 谱方法参数
  dealias = True  # 使用2/3去混淆规则
  if dealias:
      k_max = nx//2 * 2//3  # 去混淆截断波数

  # 初始化波数
  k = jnp.fft.fftfreq(nx, d=1.0/nx) * 2*jnp.pi/L
  k = jnp.fft.fftshift(k)

  @jax.jit
  def spectral_step(u, dt):
      """伪谱方法时间步进 (使用4阶Runge-Kutta)"""
      # 傅里叶变换
      u_hat = jnp.fft.fft(u)
      
      def rhs(u_hat):
          # 反变换得到物理空间速度
          u = jnp.fft.ifft(u_hat)
          # 计算非线性项
          if dealias:
              u_hat_trunc = jnp.fft.fftshift(u_hat)
              u_hat_trunc = jnp.where(jnp.abs(k) > k_max, 0, u_hat_trunc)
              u_hat_trunc = jnp.fft.ifftshift(u_hat_trunc)
              u = jnp.fft.ifft(u_hat_trunc)
          nonlinear = 0.5 * u**2
          # 导数的谱计算
          nonlinear_hat = jnp.fft.fft(nonlinear)
          return -1j * k * nonlinear_hat
      
      # RK4时间积分
      k1 = rhs(u_hat)
      k2 = rhs(u_hat + 0.5*dt*k1)
      k3 = rhs(u_hat + 0.5*dt*k2)
      k4 = rhs(u_hat + dt*k3)
      
      u_hat_new = u_hat + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
      return u_hat_new

  def simulate_spectral(u0, nt, dt):
      u_hat = jnp.fft.fft(u0)
      def body_fn(i, u_hat):
          return spectral_step(u_hat, dt)
      u_hat_final = jax.lax.fori_loop(0, nt, body_fn, u_hat)
      return jnp.fft.ifft(u_hat_final).real

  # 运行模拟
  u_spectral = simulate_spectral(u0, nt, dt)

  # 守恒性验证
  mass_spectral = jnp.sum(u_spectral) * dx
  print(f"Spectral质量守恒误差: {abs(mass_spectral - mass_initial):.2e}")

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
