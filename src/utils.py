import argparse
import copy

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsla
import numpy as np
import numpy.linalg as nalg
import numpy.random as r
from matplotlib import cm
from matplotlib import pyplot as plt

jax.config.update('jax_enable_x64', True)

def read_data(filename=None):

  data = np.load(filename)
  U = torch.from_numpy(data['U']).to(torch.float32)
  label = torch.from_numpy(data['label']).to(torch.float32)
  arg = data['arg']
  return arg, U, label


def parsing():
  parser = argparse.ArgumentParser(description='manual to this script')
  parser.add_argument('--type', type=str, default='RD')
  parser.add_argument('--gamma', type=float, default=0.05)
  parser.add_argument('--Re', type=int, default=400)
  parser.add_argument('--n', type=int, default=64)
  parser.add_argument('--batch_size', type=int, default=1000)
  parser.add_argument('--GPU', type=int, default=0)
  args = parser.parse_args()
  gamma = args.gamma
  Re = args.Re
  n = args.n
  type = args.type
  GPU = args.GPU
  if type == 'RD':
    ds_parameter = int(gamma * 20)
  else:
    ds_parameter = Re
  return n, gamma, Re, type, GPU, ds_parameter


def preprocessing(arg, simutype, U, label, device, flag=True):
  # later we can combine this function with the read_data function by
  # including all the parameters into the .npz file
  nx, ny, dt, T, label_dim = arg
  nx = int(nx)
  ny = int(ny)
  label_dim = int(label_dim)
  if flag and simutype == 'NS':
    U = U[:, :, :, 1:-1, 1:-1]
  traj_num = U.shape[0]
  step_num = U.shape[1]
  label = label.to(device)
  return nx, ny, dt, T, label_dim, traj_num, step_num, U, label


############################################################################################################
#                   Numerical solver of the reaction-diffusion equation:
# For linear term, we use different discretization scheme, e.g. explicit, implicit, Crank-Nielson, ADI etc.
# For nonlinear term, we use the explicit scheme (need to implement Ashford)
############################################################################################################


def assembly_RDmatrix(n, dt, dx, beta=1.0, gamma=0.05, d=2):
  """assemble matrices used in the calculation
    A1 = I - gamma dt \Delta, used in implicit discretization of diffusion term, size n2*n2
    A2 = I - gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    A3 = I + gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    D, size 4n2*n2, Jacobi of the Newton solver in CN discretization
    :d: ratio between the diffusion coeff for u & v
    """

  global L_uminus, L_uplus, L_vminus, L_vplus, A_uplus, A_uminus, A_vplus, A_vminus, \
            A0_u, A0_v, L, L2

  L = jnp.eye(n) * -2 + jnp.eye(n, k=1) + jnp.eye(n, k=-1)
  L = L.at[0, -1].set(1)
  L = L.at[-1, 0].set(1)
  L = L / (dx**2)
  L2 = jnp.kron(L, jnp.eye(n)) + jnp.kron(jnp.eye(n), L)

  # matrix for ADI scheme
  L_uminus = jnp.eye(n) - L * gamma * dt / 2
  L_uplus = jnp.eye(n) + L * gamma * dt / 2
  L_vminus = jnp.eye(n) - L * gamma * dt / 2 * d
  L_vplus = jnp.eye(n) + L * gamma * dt / 2 * d

  A0_u = jnp.eye(n * n) + L2 * gamma * dt
  A0_v = jnp.eye(n * n) + L2 * gamma * dt * d
  A_uplus = jnp.eye(n * n) + L2 * gamma * dt / 2
  A_uminus = jnp.eye(n * n) - L2 * gamma * dt / 2
  A_vplus = jnp.eye(n * n) + L2 * gamma * dt / 2 * d
  A_vminus = jnp.eye(n * n) - L2 * gamma * dt / 2 * d
  # L = spa.csc_matrix(L)
  # L_uminus = spa.csc_matrix(L_uminus)
  # L_uplus = spa.csc_matrix(L_uplus)
  # L_vminus = spa.csc_matrix(L_vminus)
  # L_vplus = spa.csc_matrix(L_vplus)
  # L2 = spa.kron(L, np.eye(n)) + spa.kron(np.eye(n), L)
  # A_uplus = spa.eye(n*n) + L2 * gamma * dt/2
  # A_uminus = spa.eye(n*n) - L2 * gamma * dt/2
  # A_vplus = spa.eye(n*n) + L2 * gamma * dt/2 * d
  # A_vminus = spa.eye(n*n) - L2 * gamma * dt/2 * d


def RD_exp(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """
    explicit forward Euler solver for FitzHugh-Nagumo RD equation
    :input:
    u, v: initial condition, shape [nx, ny], different nx, ny is used for different diffusion coeff
    """

  nx = u.shape[0]
  ny = u.shape[1]
  u_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  v_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  rhsu_ = jnp.zeros([nx, ny])
  rhsv_ = jnp.zeros([nx, ny])
  if jnp.linalg.norm(source) != 0:
    rhsu_ = source[0].reshape(nx * ny)
    rhsv_ = source[1].reshape(nx * ny)
  u = u.reshape(nx * ny)
  v = v.reshape(nx * ny)

  for i in range(step_num):
    tmpu = A0_u @ u + dt * (u - v - u**3 + alpha) + rhsu_ * dt
    tmpv = A0_v @ v + beta * dt * (u - v) + rhsv_ * dt
    u = tmpu
    v = tmpv

    if (i + 1) % writeInterval == 0:
      u_hist = u_hist.at[(i - 0) // writeInterval].set(u.reshape(nx, ny))
      v_hist = v_hist.at[(i - 0) // writeInterval].set(v.reshape(nx, ny))

  return u_hist, v_hist


def RD_semi(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """semi-implicit solver for FitzHugh-Nagumo RD equation"""

  nx = u.shape[0]
  ny = u.shape[1]
  u_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  v_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  rhsu_ = jnp.zeros([nx, ny])
  rhsv_ = jnp.zeros([nx, ny])
  if jnp.linalg.norm(source) != 0:
    rhsu_ = source[0].reshape(nx * ny)
    rhsv_ = source[1].reshape(nx * ny)
  u = u.reshape(nx * ny)
  v = v.reshape(nx * ny)

  for i in range(step_num):
    rhsu = A_uplus @ u + dt * (u - v - u**3 + alpha) + rhsu_ * dt
    rhsv = A_vplus @ v + beta * dt * (u - v) + rhsv_ * dt
    u, _ = jsla.cg(A_uminus, rhsu)
    v, _ = jsla.cg(A_vminus, rhsv)

    if (i + 1) % writeInterval == 0:
      u_hist = u_hist.at[(i - 0) // writeInterval].set(u.reshape(nx, ny))
      v_hist = v_hist.at[(i - 0) // writeInterval].set(v.reshape(nx, ny))

  return u_hist, v_hist

def RD_adi(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """ADI solver for FitzHugh-Nagumo RD equation"""

  u = jnp.array(u)
  v = jnp.array(v)
  @jax.jit
  def update(u, v):

    rhsu = rhsu_ * dt + L_uplus @ u @ L_uplus + dt * (u - v - u**3 + alpha)
    rhsv = rhsv_ * dt + L_vplus @ v @ L_vplus + beta * dt * (u - v)

    u = jax.scipy.linalg.solve(L_uminus, rhsu)
    u = jax.scipy.linalg.solve(L_uminus, u.T)
    u = u.T
    v = jax.scipy.linalg.solve(L_vminus, rhsv)
    v = jax.scipy.linalg.solve(L_vminus, v.T)
    v = v.T
    return u, v

  nx = u.shape[0]
  ny = u.shape[1]
  u_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  v_hist = jnp.zeros([step_num // writeInterval, nx, ny])
  rhsu_ = jnp.zeros([nx, ny])
  rhsv_ = jnp.zeros([nx, ny])
  if jnp.linalg.norm(source) != 0:
    rhsu_ = source[0].reshape(nx, ny)
    rhsv_ = source[1].reshape(nx, ny)
  flag = True

  for i in range(step_num):
    u, v = update(u, v)
    if jnp.any(jnp.isnan(u)) or jnp.any(jnp.isnan(v)) or jnp.any(jnp.isinf(u)) or jnp.any(jnp.isinf(v)):
      flag = False
      break

    if (i + 1) % writeInterval == 0:
      u_hist = u_hist.at[(i - 0) // writeInterval, :].set(u)
      v_hist = v_hist.at[(i - 0) // writeInterval, :].set(v)

  return u_hist, v_hist, flag


def RD_cn(
  u: jnp.ndarray,
  v: jnp.ndarray,
  dt: float,
  source: jnp.ndarray = 0,
  alpha: float = .01,
  beta: float = 1.0,
  step_num: int = 200,
  writeInterval: int = 1
):
  """full implicit solver with Crank-Nielson discretization
    TODO: we have not tested this function yet"""

  global L, D_
  dt = 1 / step_num
  t_array = np.array([5, 10, 20, 40, 80])

  #t_array = np.array([1, 2, 3, 4, 80])

  #plt.subplot(231)
  #plt.imshow(u.reshape(n, n), cmap = cm.jet)

  def F(u_next, v_next, u, v):
    Fu = A2 @ u_next - A3 @ u + (
      u_next**3 + u**3 + v_next + v - u_next - u - alpha
    ) * dt / 2
    Fv = A2 @ v_next - A3 @ v + (v_next + v - u_next - u) * dt * beta / 2
    res = np.hstack([Fu, Fv])
    return res

  def Newton(n):

    global u, v, L, D_
    # we use the semi-implicit scheme iteration as the initial guess of Newton method
    rhsu = u + dt * (u - v + u**3 + alpha)
    rhsv = v + beta * dt * (u - v)
    u_next = sps(A1, rhsu)
    v_next = sps(A1, rhsv)
    res = F(u_next, v_next, u, v)

    count = 0
    while nalg.norm(res) > tol:
      D_[:n * n, :n *
         n] = D_[:n * n, :n *
                 n] + dt / 2 * (spa.diags(3 * (u_next**2)) - spa.eye(n * n))
      D = D_.tocsr()
      #duv = sps(D, res)
      # GMRES with initial guess pressure in last time step
      duv = jsla.gmres(A=D, b=res.reshape(n * n), x0=duv).reshape([n, n])
      # BiSTABCG with initial guess pressure in last time step
      duv = jsla.bicgstab(A=D, b=res.reshape(n * n), x0=duv).reshape([n, n])
      u_next = u_next - duv[:n * n]
      v_next = v_next - duv[n * n:]
      res = F(u_next, v_next, u, v)
      count = count + 1
      print(scalg.norm(res))
    print(count)

    u = u_next
    v = v_next

  for i in range(step_num):
    for j in range(5):
      #if i == t_array[j] * step_num / 100:
      #    plt.subplot(2, 3, j+2)
      #    plt.imshow(u.reshape(n, n), cmap = cm.jet)
      #    plt.colorbar()
      Newton()

  #plt.show()
  return u, v


def assembly_NSmatrix(nx, ny, dt, dx, dy):
  """assemble matrices used in the calculation
    LD: Laplacian operator with Dirichlet BC
    LN: Laplacian operator with Neuman BC, notice that this operator may have different form 
        depends on the position of the boundary, here we use the case that boundary is between 
        the outmost two grids
    L:  Laplacian operator associated with current BC with three Neuman BCs on upper, lower, left boundary and a Dirichlet BC on right
    """

  global L
  LNx = np.eye(nx) * (-2)
  LNy = np.eye(ny) * (-2)
  for i in range(1, nx - 1):
    LNx[i, i - 1] = 1
    LNx[i, i + 1] = 1
  for i in range(1, ny - 1):
    LNy[i, i - 1] = 1
    LNy[i, i + 1] = 1
  LNx[0, 1] = 1
  LNx[0, 0] = -1
  LNx[-1, -1] = -1
  LNx[-1, -2] = 1
  LNy[0, 1] = 1
  LNy[0, 0] = -1
  LNy[-1, -1] = -1
  LNy[-1, -2] = 1
  LNx = spa.csc_matrix(LNx / (dx**2))
  LNy = spa.csc_matrix(LNy / (dy**2))
  # BE CAREFUL, SINCE THE LAPLACIAN MATRIX IN X Y DIRECTION IS NOT THE SAME
  #L2N = spa.kron(LNy, spa.eye(nx)) + spa.kron(spa.eye(ny), LNx)
  L2N = spa.kron(LNx, spa.eye(ny)) + spa.kron(spa.eye(nx), LNy)
  L = copy.deepcopy(L2N)
  #for i in range(ny):
  #    L[(i+1)*nx - 1, (i+1)*nx - 1] = L[(i+1)*nx - 1, (i+1)*nx - 1] - 2
  for i in range(ny):
    L[-1 - i, -1 - i] = L[-1 - i, -1 - i] - 2 / (dx**2)

  return


def projection_correction(
  u,
  v,
  t,
  dx=1 / 32,
  dy=1 / 32,
  nx=128,
  ny=32,
  y0=0.325,
  eps=1e-7,
  dt=.01,
  Re=100,
  flag=True
):
  """projection method to solve the incompressible NS equation
    The convection discretization is given by central difference
    u_ij (u_i+1,j - u_i-1,j)/2dx + \Sigma v_ij (u_i,j+1 - u_i,j-1)/2dx"""

  # central difference for first derivative
  u_x = (u[2:, 1:-1] - u[:-2, 1:-1]) / dx / 2
  u_y = (u[1:-1, 2:] - u[1:-1, :-2]) / dy / 2
  v_x = (v[2:, 1:-1] - v[:-2, 1:-1]) / dx / 2
  v_y = (v[1:-1, 2:] - v[1:-1, :-2]) / dy / 2

  # five pts scheme for Laplacian
  u_xx = (-2 * u[1:-1, 1:-1] + u[2:, 1:-1] + u[:-2, 1:-1]) / (dx**2)
  u_yy = (-2 * u[1:-1, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]) / (dy**2)
  #u_xy = (u[2:,2:]+u[:-2,:-2]-2*u[1:-1,1:-1])/(dx**2)/2 - \
  #        (u_xx+u_yy)/2
  v_xx = (-2 * v[1:-1, 1:-1] + v[2:, 1:-1] + v[:-2, 1:-1]) / (dx**2)
  v_yy = (-2 * v[1:-1, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2]) / (dy**2)
  #v_xy = (v[2:,2:]+v[:-2,:-2]-2*v[1:-1,1:-1])/(dx**2)/2 - \
  #        (v_xx+v_yy)/2

  # interpolate u, v on v, u respectively, we interpolate using the four neighbor nodes
  u2v = (u[:-2, 1:-2] + u[1:-1, 1:-2] + u[:-2, 2:-1] + u[1:-1, 2:-1]) / 4
  v2u = (v[1:-1, :-1] + v[2:, :-1] + v[1:-1, 1:] + v[2:, 1:]) / 4

  # prediction step: forward Euler
  u[1:-1, 1:-1] = u[
    1:-1, 1:-1] + dt * ((u_xx + u_yy) / Re - u[1:-1, 1:-1] * u_x - v2u * u_y)
  v[1:-1, 1:-1] = v[
    1:-1, 1:-1] + dt * ((v_xx + v_yy) / Re - u2v * v_x - v[1:-1, 1:-1] * v_y)

  # correction step: calculating the residue of Poisson equation as the divergence of new velocity field
  divu = (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, :-1]) / dy
  # GMRES with initial guess pressure in last time step
  p = jsla.gmres(A=L, b=divu.reshape(nx * ny), x0=p).reshape([nx, ny])
  # BiSTABCG with initial guess pressure in last time step
  p = jsla.bicgstab(A=L, b=divu.reshape(nx * ny), x0=p).reshape([nx, ny])

  u[1:-2, 1:-1] = u[1:-2, 1:-1] - (p[1:, :] - p[:-1, :]) / dx
  v[1:-1, 1:-1] = v[1:-1, 1:-1] - (p[:, 1:] - p[:, :-1]) / dy
  u[-2, 1:-1] = u[-2, 1:-1] + 2 * p[-1, :] / dx

  # check the corrected velocity field is divergence free
  divu = (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, :-1]) / dy
  if flag and nalg.norm(divu) > eps:
    print(nalg.norm(divu))
    print(t)
    print("Velocity field is not divergence free!!!")
    flag = False

  # update Dirichlet BC on left, upper, lower boundary
  u[:, 0] = -u[:, 1]
  u[:, -1] = -u[:, -2]
  v[0, 1:-1] = 2 * np.exp(-50 * (np.linspace(dy, 1 - dy, ny - 1) - y0)**2
                          ) * np.sin(t) - v[1, 1:-1]
  # update Neuman BC on right boundary
  u[-1, :] = u[-3, :]
  v[-1, :] = v[
    -2, :]  # alternative choice to use Neuman BC for v on the right boundary
  #v[-1, 1:-1] = v[-1, 1:-1] + (p[-1, 1:] - p[-1, :-1])/dy

  return u, v, p / dt, flag
