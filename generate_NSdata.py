import utils
import numpy as np
import numpy.random as r
import copy
import argparse


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--Re', type=int, default=100)
parser.add_argument('--nx', type=int, default=128)
#parser.add_argument('--dt', type=float, default=0.001)
#parser.add_argument('--T', type=int, default=10000)
args = parser.parse_args()
Re = args.Re
nx = args.nx
#dt = args.dt
#T = args.T
print('Generating NS data with Re = '+str(Re)+', nx = '+str(nx)+'...')


dim = 2                     # dimension of the problem
ny = nx//4
dx = 1/ny
dy = 1/ny
y0 = 0.325
u = np.zeros([nx+2, ny+2])
v = np.zeros([nx+2, ny+1])
p = np.zeros([nx, ny])      # staggered grid, the size of grid p is undetermined
divu = np.zeros([nx, ny])   # source term in poisson equation: divergence of the predicted velocity field
eps = 1e-7
dt = .01
T = 10


utils.assembly_NSmatrix(nx, ny, dt, dx, dy)
debug = False
flag = False
u = np.zeros([nx+2, ny+2])
#u_hist1 = np.zeros([5, nx+2, ny+2])
v = np.zeros([nx+2, ny+1])
p = np.zeros([nx, ny])                # staggered grid, the size of grid p is undetermined
u[0, 1:-1] = np.exp(-50*(np.linspace(dy/2, 1-dy/2, ny) - y0)**2)
#v[0, 1:-1] = np.exp(-50*(np.linspace(1/nx, 1-1/nx, nx-1) - y0)**2)
u_hist = np.zeros([2000, nx+2, ny+2])
v_hist = np.zeros([2000, nx+2, ny+1])
p_hist = np.zeros([2000, nx, ny])


T = 20
t_array = np.array([.5, 1, 2, 4, 8, 16])
for i in range(int(T/dt)):
    if flag:
        break
    t = i*dt
    u, v = utils.projection_method(u, v, t, dx=dx, dy=dy, nx=nx, ny=ny, y0=y0, eps=eps, dt=dt, Re=Re)
    u_hist[i, :, :] = u
    v_hist[i, :, :] = v
    p_hist[i, :, :] = p       


u = copy.deepcopy(u_hist)
v = copy.deepcopy(v_hist)
p = copy.deepcopy(p_hist)
print(u.shape)
print(v.shape)
print(p.shape)
np.savez('../data/NS/nx'+str(nx)+'Re'+str(Re)+'.npz', u=u, v=v, label=p)