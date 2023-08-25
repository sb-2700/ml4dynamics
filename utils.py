import torch
import numpy as np
import numpy.linalg as nalg
import numpy.random as r
import scipy.linalg as scalg
import scipy.sparse as spa
import torch.nn as nn
import copy
import pdb


from scipy.sparse.linalg import spsolve as sps


def read_data(filename=None):


    data = np.load(filename)
    u = torch.from_numpy(data['u']).to(torch.float32)
    v = torch.from_numpy(data['v']).to(torch.float32)
    label = torch.from_numpy(data['label']).to(torch.float32)
    arg = data['arg']
    return arg, u, v, label


'''
def read_data(i=0):
    
    
    global u64, v64, label64, n, step_num
    data_size = 1
    u64 = np.load('data/u64-200.npy')
    u64 = torch.from_numpy(u64)
    u64 = u64.to(torch.float32)
    u64 = u64[i, :, :]
    v64 = np.load('data/v64-200.npy')
    v64 = torch.from_numpy(v64)
    v64 = v64.to(torch.float32)
    v64 = v64[i, :, :]
    labelu64 = np.load('data/labelu64-10.npy')
    labelu64 = torch.from_numpy(labelu64)
    labelu64 = labelu64.to(torch.float32)
    labelu64 = labelu64[i, :, :, :]
    labelv64 = np.load('data/labelv64-10.npy')
    labelv64 = torch.from_numpy(labelv64)
    labelv64 = labelv64.to(torch.float32)
    labelv64 = labelv64[i, :, :, :]
    label64 = torch.zeros(step_num, 2, n, n)
    label64[:,0,:,:] = labelu64
    label64[:,1,:,:] = labelv64
    
    
    
    label = torch.zeros(100, 2, n, n)
    for i in range(u64.shape[0]):
        label[i, 0, :, :] = (u64[i, :] - u64[i, :]**3 - v64[i, :] + alpha).reshape([n, n])
        label[i, 1, :, :] = beta * (u64[i, :] - v64[i, :]).reshape([n, n])
    label = label.to(torch.float32)
    
    
    case_num = 40
    if data_size == 0:
        # randomly sample data from the total distribution
        case_num = 10
        tmp_x = torch.zeros([step_num, state_dim, case_num])
        tmp_u = torch.zeros([step_num, case_num])
        for i in range(case_num):
            tmp = int(np.floor(r.rand() * traj_num))
            tmp_x[:, :, i] = x[:, :, tmp].clone()
            tmp_u[:, i] = u[:, tmp].clone()
        x = tmp_x.clone()
        u = tmp_u.clone()'''


def assembly_RDmatrix(n, dt, dx, beta, gamma):
    """assemble matrices used in the calculation
    A1 = I - gamma dt \Delta, used in implicit discretization of diffusion term, size n2*n2
    A2 = I - gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    A3 = I + gamma dt/2 \Delta, used in CN discretization of diffusion term, size n2*n2
    D, size 4n2*n2, Jacobi of the Newton solver in CN discretization
    """
    
    
    global A0, A1, A2, A3, D_
    L = np.eye(n) * (-2)
    for i in range(1, n-1):
        L[i, i-1] = 1
        L[i, i+1] = 1
    L[0, 1] = 1
    L[0, -1] = 1
    L[-1, 0] = 1
    L[-1, -2] = 1
    L = L/(dx**2)
    L = spa.csc_matrix(L)
    L2 = spa.kron(L, np.eye(n)) + spa.kron(np.eye(n), L)
    A0 = spa.eye(n*n) + L2 * gamma * dt 
    A1 = spa.eye(n*n) - L2 * gamma * dt             
    A2 = spa.eye(n*n) - L2 * gamma * dt/2            
    A3 = spa.eye(n*n) + L2 * gamma * dt/2         
    
    
    D_ = spa.lil_matrix((2*n*n, 2*n*n))
    D_[:n*n, :n*n] = A2                                                # dF_u/du
    D_[n*n:, n*n:] = A2 + dt*beta*spa.eye(n*n)/2                       # dF_v/dv
    D_[:n*n, n*n:] = dt*spa.eye(n*n)/2                                 # dF_u/dv
    D_[n*n:, :n*n] = -dt*beta*spa.eye(n*n)/2                           # dF_v/du


def RD_exp():
    """explicit forward Euler solver for FitzHugh-Nagumo RD equation"""
    
    
    global u, v, L, u_hist, v_hist, step_num, alpha, beta
    dt = 1/step_num
    t_array = np.array([5, 10, 20, 40, 80])
    
    
    #plt.subplot(231)
    #plt.imshow(u.reshape(n, n), cmap = cm.jet)
    
    
    for i in range(step_num):
        for j in range(5):
            #if i == t_array[j] * step_num / 100:
            #    plt.subplot(2, 3, j+2)
            #    plt.imshow(u.reshape(n, n), cmap = cm.jet)
            #    plt.colorbar()
            #tmpu = A0 @ u + dt * (u - v + u**3 + alpha)
            tmpu = A0 @ u + dt * (u - v - u**3 + alpha)
            tmpv = A0 @ v + beta * dt * (u - v)
            u = tmpu
            v = tmpv
            u_hist[i, :] = u
            v_hist[i, :] = v
        
     
    #plt.colorbar()
    #plt.show()
    return u, v
    
    
def RD_semi(u, v, alpha=.01, beta=.2, gamma=.05, step_num=200, plot=True, write=True):
    """semi-implicit solver for FitzHugh-Nagumo RD equation"""
    
    
    global L, u_hist, v_hist
    dt = 1/step_num
    t_array = np.array([5, 10, 20, 40, 80])
    u_hist = np.zeros([step_num, u.size])
    v_hist = np.zeros([step_num, v.size])
    
    
    #if plot:
    #    plt.subplot(231)
    #    plt.imshow(u.reshape(n, n), cmap = cm.jet)
    
    
    for i in range(step_num):
        #for j in range(5):
            #if i == t_array[j] * step_num / 100:
            #    if plot:
            #        plt.subplot(2, 3, j+2)
            #        plt.imshow(u.reshape(n, n), cmap = cm.jet)
            #        plt.colorbar()
        rhsu = u + dt * (u - v + u**3 + alpha)
        rhsv = v + beta * dt * (u - v)
        u = sps(A1, rhsu)
        v = sps(A1, rhsv)
        if write:
            u_hist[i, :] = u
            v_hist[i, :] = v
        elif (i+1)%10 == 0:
            u_hist[(i-0)//10, :] = u
            v_hist[(i-0)//10, :] = v
            
    
    #if plot:
        #plt.colorbar()
    #    plt.show()
    return u_hist, v_hist
    

def RD_cn():
    """full implicit solver with Crank-Nielson discretization"""
    
    
    global u, v, L, D_, step_num, alpha, beta, gamma, tol
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
    
    
    global divu, L, p
    
    
    #if 'L' in locals():
    #    print('L is a local variable')
    #if 'L' in globals():
    #    print('L is a global variable')
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


    return u, v, flag