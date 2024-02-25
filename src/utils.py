import os
import copy
import argparse
import pdb
import torch
import numpy as np
import numpy.linalg as nalg
import numpy.random as r
import jax.scipy.sparse.linalg as jsla
from matplotlib import pyplot as plt
from matplotlib import cm

from scipy.sparse.linalg import spsolve as sps

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
        ds_parameter = int(gamma*20)
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


def assembly_RDmatrix(n, dt, dx, beta=1.0, gamma=0.05):
    """assemble matrices used in the calculation
    A1 = I - gamma dt \Delta, used in implicit discretization of diffusion term, size n2*n2
    A2 = I - gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    A3 = I + gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    D, size 4n2*n2, Jacobi of the Newton solver in CN discretization
    """
    
    global L_uminus, L_uplus, L_vminus, L_vplus, A_uplus, A_uminus, A_vplus, A_vminus
    L = np.eye(n) * (-2)
    d = 2                                       # ratio between the diffusion coeff for u & v
    for i in range(1, n-1):
        L[i, i-1] = 1
        L[i, i+1] = 1
    L[0, 1] = 1
    L[0, -1] = 1
    L[-1, 0] = 1
    L[-1, -2] = 1
    L = L/(dx**2)
    L_uminus = np.eye(n) - L * gamma * dt/2
    L_uplus = np.eye(n) + L * gamma * dt/2
    L_vminus = np.eye(n) - L * gamma * dt/2 * d
    L_vplus = np.eye(n) + L * gamma * dt/2 * d
    L = spa.csc_matrix(L)
    L_uminus = spa.csc_matrix(L_uminus)
    L_uplus = spa.csc_matrix(L_uplus)
    L_vminus = spa.csc_matrix(L_vminus)
    L_vplus = spa.csc_matrix(L_vplus)
    L2 = spa.kron(L, np.eye(n)) + spa.kron(np.eye(n), L) 
    A_uplus = spa.eye(n*n) + L2 * gamma * dt/2           
    A_uminus = spa.eye(n*n) - L2 * gamma * dt/2
    A_vplus = spa.eye(n*n) + L2 * gamma * dt/2 * d           
    A_vminus = spa.eye(n*n) - L2 * gamma * dt/2 * d                    
    
    '''D_ = spa.lil_matrix((2*n*n, 2*n*n))
    D_[:n*n, :n*n] = A2                                                # dF_u/du
    D_[n*n:, n*n:] = A2 + dt*beta*spa.eye(n*n)/2                       # dF_v/dv
    D_[:n*n, n*n:] = dt*spa.eye(n*n)/2                                 # dF_u/dv
    D_[n*n:, :n*n] = -dt*beta*spa.eye(n*n)/2                           # dF_v/du'''

def RD_exp(u, v, dt, alpha=.01, beta=1.0, gamma=.05, step_num=200, plot=True, write=True):
    """
    explicit forward Euler solver for FitzHugh-Nagumo RD equation
    Modification has to be made for it to simulate PDE with different diffusion coefficient for two components
    """
    
    u_hist = np.zeros([step_num, u.size])
    v_hist = np.zeros([step_num, v.size])
    
    for i in range(step_num):
        for j in range(5):
            tmpu = A0 @ u + dt * (u - v - u**3 + alpha)
            tmpv = A0 @ v + beta * dt * (u - v)
            u = tmpu
            v = tmpv
            u_hist[i, :] = u
            v_hist[i, :] = v
        
    return u_hist, v_hist
       
def RD_semi(u, v, dt, alpha=.01, beta=1.0, gamma=.05, step_num=200, plot=True, write=True):
    """semi-implicit solver for FitzHugh-Nagumo RD equation"""
    
    global A_uplus, A_uminus, A_vplus, A_vminus
    u_hist = np.zeros([step_num, u.size])
    v_hist = np.zeros([step_num, v.size])
    
    for i in range(step_num):
        rhsu = A_uplus @ u + dt * (u - v - u**3 + alpha)
        rhsv = A_vplus @ v + beta * dt * (u - v)
        u = sps(A_uminus, rhsu)
        v = sps(A_vminus, rhsv)
        if write:
            u_hist[i, :] = u
            v_hist[i, :] = v
        elif (i+1)%10 == 0:
            u_hist[(i-0)//10, :] = u
            v_hist[(i-0)//10, :] = v
    return u_hist, v_hist
    
def RD_adi(u, v, dt, alpha=.01, beta=1.0, gamma=.05, step_num=200, writeInterval=1, plot=True, write=True):
    """ADI solver for FitzHugh-Nagumo RD equation"""
    
    global L_uminus, L_uplus, L_vminus, L_vplus
    upper_bound = 1e5
    flag = True
    u_hist = np.zeros([step_num//writeInterval, u.shape[0], u.shape[1]])
    v_hist = np.zeros([step_num//writeInterval, u.shape[0], u.shape[1]])

    for i in range(step_num):
        if nalg.norm(u) > upper_bound:
            print('unacceptable field!')
            flag = False
            break
        rhsu = L_uplus @ u @ L_uplus + dt * (u - v - u**3 + alpha)
        rhsv = L_vplus @ v @ L_vplus + beta * dt * (u - v)
        u = sps(L_uminus, rhsu)
        u = sps(L_uminus, u.T)
        u = u.T
        v = sps(L_vminus, rhsv)
        v = sps(L_vminus, v.T)
        v = v.T

        if (i+1)%writeInterval == 0:
            u_hist[(i-0)//writeInterval, :] = u
            v_hist[(i-0)//writeInterval, :] = v

    return u_hist, v_hist, flag

def RD_cn(u, v, dt, alpha=.01, beta=.2, gamma=.05, step_num=200, tol=1e-8, writeInterval=1, plot=True, write=True):
    """full implicit solver with Crank-Nielson discretization"""
    
    global L, D_
    dt = 1/step_num
    t_array = np.array([5, 10, 20, 40, 80])
    #t_array = np.array([1, 2, 3, 4, 80])
    
    
    #plt.subplot(231)
    #plt.imshow(u.reshape(n, n), cmap = cm.jet)
    
    
    def F(u_next, v_next, u, v):
        Fu = A2 @ u_next - A3 @ u + (u_next**3 + u**3 + v_next + v - u_next - u - alpha ) * dt/2
        Fv = A2 @ v_next - A3 @ v + (v_next + v - u_next - u) * dt * beta/2
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
            D_[:n*n, :n*n] = D_[:n*n, :n*n] + dt/2*(spa.diags(3*(u_next**2)) - spa.eye(n*n))
            D = D_.tocsr()
            duv = sps(D, res)
            u_next = u_next - duv[:n*n]
            v_next = v_next - duv[n*n:]
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
    for i in range(1, nx-1):
        LNx[i, i-1] = 1
        LNx[i, i+1] = 1
    for i in range(1, ny-1):
        LNy[i, i-1] = 1
        LNy[i, i+1] = 1
    LNx[0, 1] = 1
    LNx[0, 0] = -1
    LNx[-1, -1] = -1
    LNx[-1, -2] = 1
    LNy[0, 1] = 1
    LNy[0, 0] = -1
    LNy[-1, -1] = -1
    LNy[-1, -2] = 1
    LNx = spa.csc_matrix(LNx/(dx**2))
    LNy = spa.csc_matrix(LNy/(dy**2))
    # BE CAREFUL, SINCE THE LAPLACIAN MATRIX IN X Y DIRECTION IS NOT THE SAME
    #L2N = spa.kron(LNy, spa.eye(nx)) + spa.kron(spa.eye(ny), LNx)
    L2N = spa.kron(LNx, spa.eye(ny)) + spa.kron(spa.eye(nx), LNy)
    L = copy.deepcopy(L2N)
    #for i in range(ny):
    #    L[(i+1)*nx - 1, (i+1)*nx - 1] = L[(i+1)*nx - 1, (i+1)*nx - 1] - 2
    for i in range(ny):
        L[-1-i, -1-i] = L[-1-i, -1-i] - 2/(dx**2)
        
    return    


def projection_method(u, v, t, dx=1/32, dy=1/32, nx=128, ny=32, y0=0.325, eps=1e-7, dt=.01, Re=100, flag=True):
    """projection method to solve the incompressible NS equation
    The convection discretization is given by central difference
    u_ij (u_i+1,j - u_i-1,j)/2dx + \Sigma v_ij (u_i,j+1 - u_i,j-1)/2dx"""
    
    # central difference for first derivative
    u_x = (u[2:,1:-1]-u[:-2,1:-1])/dx/2
    u_y = (u[1:-1,2:]-u[1:-1,:-2])/dy/2
    v_x = (v[2:,1:-1]-v[:-2,1:-1])/dx/2
    v_y = (v[1:-1,2:]-v[1:-1,:-2])/dy/2
    
    # five pts scheme for Laplacian
    u_xx = (-2*u[1:-1,1:-1] + u[2:,1:-1] + u[:-2,1:-1])/(dx**2)
    u_yy = (-2*u[1:-1,1:-1] + u[1:-1,2:] + u[1:-1,:-2])/(dy**2)
    #u_xy = (u[2:,2:]+u[:-2,:-2]-2*u[1:-1,1:-1])/(dx**2)/2 - \
    #        (u_xx+u_yy)/2
    v_xx = (-2*v[1:-1,1:-1] + v[2:,1:-1] + v[:-2,1:-1])/(dx**2)
    v_yy = (-2*v[1:-1,1:-1] + v[1:-1,2:] + v[1:-1,:-2])/(dy**2)
    #v_xy = (v[2:,2:]+v[:-2,:-2]-2*v[1:-1,1:-1])/(dx**2)/2 - \
    #        (v_xx+v_yy)/2
    
    # interpolate u, v on v, u respectively, we interpolate using the four neighbor nodes
    u2v = (u[:-2, 1:-2] + u[1:-1, 1:-2] + u[:-2, 2:-1] + u[1:-1, 2:-1])/4
    v2u = (v[1:-1, :-1] + v[2:, :-1] + v[1:-1, 1:] + v[2:, 1:])/4
    
    
    # prediction step: forward Euler 
    u[1:-1,1:-1] = u[1:-1,1:-1] + dt * ((u_xx + u_yy)/Re - u[1:-1,1:-1] * u_x - v2u * u_y)
    v[1:-1,1:-1] = v[1:-1,1:-1] + dt * ((v_xx + v_yy)/Re - u2v * v_x - v[1:-1,1:-1] * v_y)
        
    
    # correction step: calculating the residue of Poisson equation as the divergence of new velocity field
    divu = (u[1:-1, 1:-1] - u[:-2, 1:-1])/dx + (v[1:-1, 1:] - v[1:-1, :-1])/dy
    p = sps(L, divu.reshape(nx*ny)).reshape([nx, ny])
    
    
    u[1:-2, 1:-1] = u[1:-2, 1:-1] - (p[1:,:] - p[:-1,:])/dx
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - (p[:,1:] - p[:,:-1])/dy
    u[-2, 1:-1] = u[-2, 1:-1] + 2 * p[-1,:]/dx
    
    
    # check the corrected velocity field is divergence free
    divu = (u[1:-1, 1:-1] - u[:-2, 1:-1])/dx + (v[1:-1, 1:] - v[1:-1, :-1])/dy
    if flag and nalg.norm(divu) > eps:
        print(nalg.norm(divu))
        print(t)
        print("Velocity field is not divergence free!!!")
        flag = False
        
        
    # update Dirichlet BC on left, upper, lower boundary
    u[:, 0] = -u[:, 1]
    u[:, -1] = -u[:, -2]
    v[0, 1:-1] = 2*np.exp(-50*(np.linspace(dy, 1-dy, ny-1) - y0)**2)*np.sin(t) - v[1, 1:-1]
    # update Neuman BC on right boundary 
    u[-1, :] = u[-3, :]
    v[-1, :] = v[-2, :]          # alternative choice to use Neuman BC for v on the right boundary
    #v[-1, 1:-1] = v[-1, 1:-1] + (p[-1, 1:] - p[-1, :-1])/dy


    return u, v, p/dt, flag