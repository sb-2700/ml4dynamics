import numpy as np
import numpy.linalg as nalg
import scipy.sparse as spa
import torch
import copy
import torch.nn as nn
from abc import ABCMeta, abstractmethod


class Simulator():
    __metaclass__ = ABCMeta

    def __init__(self, model, ed_model, device, u_hist, v_hist, step_num=1000, dt=0.001, n=64, n2=32, type=1, img_size=64):
        # this n and n2 is originally for coarse-fine grid
        self.step_num = step_num
        self.dt = dt
        self.n = n
        self.dx = 6.4/n
        self.n2 = n2
        self.dx2 = 6.4/n2
        self.model = model
        self.ed_model = ed_model
        self.device = device
        self.type = type
        if u_hist.shape[0] != self.step_num:
            print("Error in set_config: Inconsistent step number!")
        if u_hist.shape[1] != self.n:
            print("Error in set_config: Inconsistent grid size!")
        self.u_hist = u_hist
        self.v_hist = v_hist
        self.u_hist_simu = copy.deepcopy(u_hist)
        self.v_hist_simu = copy.deepcopy(v_hist)
        self.u = self.u_hist[0, :, :]
        self.v = self.v_hist[0, :, :]
        self.ds_hist = np.zeros(step_num)
        self.error_hist = np.zeros(step_num)
        self.div_hist = np.zeros(step_num)
        self.criterion = nn.MSELoss()               # add this parameter to the parameter lists
        self.img_size = img_size
        self.fluid_type = 'None'


    @abstractmethod
    def outer_step(self):
        print("abstract method!")


    @abstractmethod
    def inner_step(self):
        print("abstract method!")


    def simulator(self, rewrite=False):
        """simulating the dynamics using the given control"""
        
        
        loss_hist = []
        if self.fluid_type == 'RD':
            t_array = np.array([10, 20, 40, 80, 160])
        elif self.fluid_type == 'NS':
            t_array = np.array([50, 100, 200, 400, 800])
        else:
            print('Unknown fluid type!!!')
            raise NameError
        
        
        for i in range(self.step_num-1):
            self.outer_step()


            uv = torch.zeros([1, 2, self.nx, self.ny], dtype=torch.float32)
            # this is for RD equation
            if self.fluid_type == 'RD':
                uv[0, 0, :, :] = torch.from_numpy(self.u)
                uv[0, 1, :, :] = torch.from_numpy(self.v)
            # this is for NS equation, it is not necessary to do this, we have to figure out a method to put these
            # two consistent, one method is to use a if sentence to distinguish two cases
            elif self.fluid_type == 'NS':
                uv[0, 0, :, :] = torch.from_numpy(self.u[1:-1,1:-1])
                uv[0, 1, :, :] = torch.from_numpy(self.v[1:-1,1:])
            else:
                print('Unknown fluid type!!!')
                raise NameError
            uv = uv.to(self.device)
            output = self.ed_model(uv)
            self.ds_hist[i] = self.criterion(uv, output).item()
            self.error_hist[i] = nalg.norm(self.u - self.u_hist[i], 'fro') + \
                                 nalg.norm(self.v - self.v_hist[i], 'fro')
            if self.fluid_type == 'NS':
                self.div_hist[i] = np.sum((self.u[1:-1, 1:-1] - self.u[:-2, 1:-1])/self.dx + (self.v[1:-1, 1:] - self.v[1:-1, :-1])/self.dx)


class RD_Simulator(Simulator):


    def __init__(self, model, ed_model, device, u_hist, v_hist, step_num=100, dt=0.01, n=64, n2=32, type=1, alpha=0.01, beta=0.25, gamma=0.05, img_size=64):
        super(RD_Simulator, self).__init__(model, ed_model, device, u_hist, v_hist, step_num, dt, n, n2, type, img_size)


        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nx = n
        self.ny = n
        self.fluid_type = 'RD'
        self.set_matrix()


    def assembly_matrix(self, n, dt, dx):
        """assemble matrices used in the calculation"""
        
        
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
        A0 = spa.eye(n*n) + L2 * self.gamma * dt 
        return A0


    def set_matrix(self):
        self.A0 = self.assembly_matrix(self.n, self.dt, self.dx)
        self.A1 = self.assembly_matrix(self.n2, self.dt, self.dx2)
        return 0


    
    def outer_step(self):
        # here we have to be careful with the deep copy and shallow copy
        if self.type == 1:
            # outer loop for first outer-inner loop structure: linear v.s. nonlinear
            uv = torch.zeros([1, 2, self.n, self.n], dtype=torch.float32)
            uv[0, 0, :, :] = torch.from_numpy(self.u)
            uv[0, 1, :, :] = torch.from_numpy(self.v)
            uv = uv.to(self.device)
            output = self.model(uv).detach().numpy()
            self.u = (self.A0 @ self.u.reshape([self.n*self.n, 1])).reshape([self.n, self.n]) + self.dt * output[0, 0, :, :]
            self.v = (self.A0 @ self.v.reshape([self.n*self.n, 1])).reshape([self.n, self.n]) + self.dt * output[0, 1, :, :]

        elif self.type == 2:
            # outer loop for second outer-inner loop structure: coarse v.s. fine grid
            # restriction from fine grid to coarse grid
            # Now the implementation only consider fine and coarse grid differs by factor 2, later can consider more
            # complex situation
            self.u_ = (self.u[::2, ::2] + self.u[::2, 1::2] + self.u[1::2, ::2] + self.u[1::2, 1::2])/4
            self.v_ = (self.v[::2, ::2] + self.v[::2, 1::2] + self.v[1::2, ::2] + self.v[1::2, 1::2])/4


            # iterate one step on coarse grid
            self.u_ = (self.A1 @ self.u_.reshape([self.n*self.n//4, 1])).reshape([self.n//2, self.n//2]) \
                        + self.dt * (self.u_ - self.v_ - self.u_**3 + self.alpha)
            self.v_ = (self.A1 @ self.v_.reshape([self.n*self.n//4, 1])).reshape([self.n//2, self.n//2]) \
                        + self.beta * self.dt * (self.u_ - self.v_)


            # interpolate from coarse grid to fine grid
            self.u[::2, ::2] = self.u_
            self.u[::2, 1::2] = self.u_
            self.u[1::2, ::2] = self.u_
            self.u[1::2, 1::2] = self.u_
            self.v[::2, ::2] = self.v_
            self.v[::2, 1::2] = self.v_
            self.v[1::2, ::2] = self.v_
            self.v[1::2, 1::2] = self.v_


            uv = torch.zeros([1, 2, self.n, self.n], dtype=torch.float32)
            uv[0, 0, :, :] = torch.from_numpy(self.u)
            uv[0, 1, :, :] = torch.from_numpy(self.v)
            uv = uv.to(self.device)
            output = self.model(uv).detach().numpy()
            self.u = self.u + output[0, 0]*self.dt
            self.u = self.u + output[0, 1]*self.dt


class NS_Simulator(Simulator):

    def __init__(self, model, ed_model, device, u_hist, v_hist, step_num=100, dt=0.01, Re=400, n=64, n2=32, type=1, nx=128, ny=32, y0=0.325, img_size=32):
        super(NS_Simulator, self).__init__(model, ed_model, device, u_hist, v_hist, step_num, dt, n, n2, type, img_size)
        self.nx = nx
        self.ny = ny
        self.y0 = y0
        self.Re = Re
        self.t = 0
        self.fluid_type = 'NS'
        self.set_matrix()


    def assembly_matrix(self, n, dt, dx):
        """assemble matrices used in the calculation"""
        
        
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
        #A0 = spa.eye(n*n) + L2 * self.gamma * dt 
        return L2


    def set_matrix(self):
        self.A0 = self.assembly_matrix(self.n, self.dt, self.dx)
        self.A1 = self.assembly_matrix(self.n2, self.dt, self.dx2)
        return 0


    def outer_step(self):
        # outer loop for first outer-inner loop structure: linear v.s. nonlinear
        # central difference for first derivative
        u = copy.deepcopy(self.u)
        v = copy.deepcopy(self.v)
        dx = self.dx
        dy = dx
        dt = self.dt
        Re = self.Re
        self.t = self.t + self.dt


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


        uv = torch.zeros([1, 2, self.nx, self.ny])
        #print(type(u[1:-1,1:-1]))
        uv[0, 0, :, :] = torch.from_numpy(u[1:-1,1:-1])
        uv[0, 1, :, :] = torch.from_numpy(v[1:-1,1:])
        uv = uv.to(self.device)
        p = self.model(uv).detach().numpy().reshape(self.nx,self.ny)
        

        u[1:-2, 1:-1] = u[1:-2, 1:-1] - (p[1:,:] - p[:-1,:])/dx
        v[1:-1, 1:-1] = v[1:-1, 1:-1] - (p[:,1:] - p[:,:-1])/dy
        u[-2, 1:-1] = u[-2, 1:-1] + 2 * p[-1,:]/dx


        # update Dirichlet BC on left, upper, lower boundary
        u[:, 0] = -u[:, 1]
        u[:, -1] = -u[:, -2]
        v[0, 1:-1] = 2*np.exp(-50*(np.linspace(dy, 1-dy, self.ny-1) - self.y0)**2)*np.sin(self.t) - v[1, 1:-1]
        # update Neuman BC on right boundary 
        u[-1, :] = u[-3, :]
        v[-1, :] = v[-2, :]          # alternative choice to use Neuman BC for v on the right boundary
        #v[-1, 1:-1] = v[-1, 1:-1] + (p[-1, 1:] - p[-1, :-1])/dy


        self.u = u
        self.v = v


        omega = 0.01
        i = int(self.t / self.dt)
        #self.u = self.u_hist[i]
        #self.v = self.v_hist[i]
        self.u = omega * self.u_hist[i] + (1-omega) * u
        self.v = omega * self.v_hist[i] + (1-omega) * v

    """
    def outer_step(self):
        # outer loop for second outer-inner loop structure: coarse v.s. fine grid
        # restriction from fine grid to coarse grid
        # Now the implementation only consider fine and coarse grid differs by factor 2, later can consider more
        # complex situation
        self.u_ = (self.u[::2, ::2] + self.u[::2, 1::2] + self.u[1::2, ::2] + self.u[1::2, 1::2])/4
        self.v_ = (self.v[::2, ::2] + self.v[::2, 1::2] + self.v[1::2, ::2] + self.v[1::2, 1::2])/4


        # iterate one step on coarse grid
        self.u_ = (self.A1 @ self.u_.reshape([self.n*self.n//4, 1])).reshape([self.n//2, self.n//2]) \
                    + self.dt * (self.u_ - self.v_ - self.u_**3 + self.alpha)
        self.v_ = (self.A1 @ self.v_.reshape([self.n*self.n//4, 1])).reshape([self.n//2, self.n//2]) \
                    + self.beta * self.dt * (self.u_ - self.v_)


        # interpolate from coarse grid to fine grid
        self.u[::2, ::2] = self.u_
        self.u[::2, 1::2] = self.u_
        self.u[1::2, ::2] = self.u_
        self.u[1::2, 1::2] = self.u_
        self.v[::2, ::2] = self.v_
        self.v[::2, 1::2] = self.v_
        self.v[1::2, ::2] = self.v_
        self.v[1::2, 1::2] = self.v_


        uv = torch.zeros([1, 2, self.n, self.n], dtype=torch.float32)
        uv[0, 0, :, :] = torch.from_numpy(self.u)
        uv[0, 1, :, :] = torch.from_numpy(self.v)
        uv = uv.to(self.device)
        output = self.model(uv).detach().numpy()
        self.u = self.u + output[0, 0]#*self.dt
        self.u = self.u + output[0, 1]#*self.dt
    """