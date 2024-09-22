import jax

import src.utils as utils

jax.config.update('jax_enable_x64', True)

n = 256
dt = .01
dx = 1/n
beta = 1
gamma = .1
utils.assembly_RDmatrix(n, dt, dx, beta, gamma)

num = 10000
from time import time
start = time()
for j in range(num):
  # u, _ = jsla.cg(L_uminus, rhsu)
  cfac = jax.scipy.linalg.cho_factor(L_uminus)
  y = jax.scipy.linalg.cho_solve(cfac, rhsu)
print('time elapsed: {:.4f}'.format(time()-start))

start = time()
for j in range(num):
  lufac = jax.scipy.linalg.lu_factor(L_uminus)
  u = jax.scipy.linalg.lu_solve(lufac, rhsu)
print('time elapsed: {:.4f}'.format(time()-start))

start = time()
for j in range(num):
  u = jax.scipy.linalg.solve(L_uminus, rhsu)
print('time elapsed: {:.4f}'.format(time()-start))

start = time()
for j in range(num):
  u = jax.scipy.linalg.solve(L_uminus, rhsu, assume_a='sym')
print('time elapsed: {:.4f}'.format(time()-start))

import jaxopt
start = time()
def matvec(x):
  return L_uminus@x
for j in range(100):
  u = jaxopt.linear_solve.solve_cg(matvec, rhsu)
print('time elapsed: {:.4f}'.format(time()-start))