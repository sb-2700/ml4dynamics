# This script plot \cref{RD-ds}
# This script plot \cref{NS-ds}
from utils import *
from simulator import *
from models import *
from test import test_simulator

def main():
    def RD_true_model(input):
        alpha = 0.01
        beta = 1.0
        label = torch.Tensor(input.shape)
        label[:,0] = input[:,0] - input[:,0]**3 - input[:,1] + alpha + 1e-4*torch.randn(input[:,0].shape)
        label[:,1] = beta*(input[:,0]- input[:,1]) + 1e-4*torch.randn(input[:,0].shape)
        return label
    
    def NS_true_model(input):
        # this model can only have batchsize one
        # central difference for first derivative

        global L
        nx = 128
        ny = 32
        u = np.zeros([nx+2, ny+2])
        v = np.zeros([nx+2, ny+1])
        u[1:-1,1:-1] = input[0,0].numpy()
        v[1:-1,1:] = input[0,1].numpy()
        u[:, 0] = -u[:, 1]
        u[:, -1] = -u[:, -2]
        v[0, 1:-1] = - v[1, 1:-1]
        # update Neuman BC on right boundary 
        u[-1, :] = u[-3, :]
        v[-1, :] = v[-2, :]
        dx = 1/32
        dy = dx
        dt = 0.02
        Re = 300
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
        p = sps(L, divu.reshape(nx*ny)).reshape([nx, ny])/dt
        return torch.from_numpy(p) + 1e-4*torch.randn(p.shape)

    def assembly_NSmatrix(nx, ny, dt, dx, dy):

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

    r.seed(0)
    case_num = 10
    simu_type = 'RD'
    n = 128
    Re = 300
    assembly_NSmatrix(nx=n, ny=n//4, dt=0.02, dx=1/32, dy=1/32)
    test_index = int(np.floor(r.rand() * case_num))
    test_index = 0
    simulator_ablation, _ = test_simulator(n=128,
                                           test_index=test_index, 
                                           simu_type=simu_type, 
                                           ds_parameter=1,
                                           model_type='true', 
                                           true_model=RD_true_model)
    simulator_ols, _ = test_simulator(n=128,
                                      test_index=test_index, 
                                      model_type='OLS', 
                                      simu_type=simu_type,
                                      ds_parameter=1)

    plt.rcParams['savefig.dpi'] = 500
    plt.rcParams['figure.dpi']  = 500 
    width = 7.5
    height = 3
    labelsize = 10
    time_scale = 500
    #################################################################
    #               Plot of the simulation result                   #
    #################################################################
    fig = plt.figure(figsize=(width, height))
    fig_num = 5
    hspace = 0.01
    wspace = -0.765
    fraction = 0.016
    pad = 0.001
    if simu_type == 'RD':
        x_left = 0
        x_right = n
        y_left = 0
        y_right = n
    if simu_type == 'NS':
        x_left = 1
        x_right = n//4+1
        y_left = 1
        y_right = -1
    t_array = (np.array([0.0, 0.2, 0.4, 0.6, 1.0])*time_scale).astype(int)
    ax1 = []
    ax2 = []
    ax3 = []
    im = []
    for i in range(fig_num):
        ax1.append(fig.add_subplot(3,fig_num,i+1))
        ax2.append(fig.add_subplot(3,fig_num,i+fig_num+1))
        ax3.append(fig.add_subplot(3,fig_num,i+fig_num*2+1))
    fig.tight_layout()
    for i in range(fig_num):
        im.append(ax1[i].imshow(simulator_ols.u_hist[t_array[i]][x_left:x_right, y_left:y_right], cmap=cm.jet))
        ax1[i].set_axis_off()
        ax2[i].imshow(simulator_ols.u_hist_simu[t_array[i]][x_left:x_right, y_left:y_right], cmap=cm.jet)
        ax2[i].set_axis_off()
        ax3[i].imshow(simulator_ablation.u_hist_simu[t_array[i]][x_left:x_right, y_left:y_right], cmap=cm.jet)
        ax3[i].set_axis_off()
    plt.subplots_adjust(hspace=hspace)
    plt.subplots_adjust(wspace=wspace)
    for i in range(fig_num):
        cbar = fig.colorbar(im[i], ax=[ax1[i], ax2[i], ax3[i]], fraction=fraction, pad=pad, orientation='horizontal')
        cbar.ax.tick_params(labelsize=5)
    plt.savefig('../../fig/exp1/{}/1.svg'.format(simu_type), dpi = 1000, bbox_inches='tight', pad_inches=0, format='svg')
    plt.savefig('../../fig/exp1/{}/1.pdf'.format(simu_type), dpi = 1000, bbox_inches='tight', pad_inches=0)
    plt.show()
    #plt.clf()


    #################################################################
    #              Plot of the ds & error comparison                #
    #################################################################
    error_hist = []
    ds_hist = []
    case_num = 1
    '''for i in range(case_num):
        simulator_ols, _ = test_simulator(n=128,
                                      test_index=test_index, 
                                      model_type='OLS', 
                                      simu_type=simu_type,
                                      ds_parameter=Re)
        error_hist.append(simulator_ols.error_hist[1:time_scale])
        ds_hist.append(np.log10(simulator_ols.ds_hist[1:time_scale]))
    error_hist_mean = np.mean(error_hist, axis=0)
    ds_hist_mean = np.mean(ds_hist, axis=0)
    error_hist_std = np.std(error_hist, axis=0)
    ds_hist_std = np.std(ds_hist, axis=0)'''
    '''fig = plt.figure(figsize=(width, height/2))
    ax = fig.add_subplot(111)
    #ax.plot(np.linspace(0, 800, time_scale), simulator_ols.error_hist[:time_scale]*10, label=r'$\left\| u(t) - \widehat u(t) \right\|_{2}^2$', color='r', linewidth=.5)
    ax.plot(simulator_ols.error_hist[1:time_scale], label=r'OLS', color='r', linewidth=.5)
    ax.plot(simulator_ablation.error_hist[1:time_scale], label=r'Ablation check', color='b', linewidth=.5)
    #ax.fill_between(error_hist_mean - error_hist_std, error_hist_mean + error_hist_std, color='blue', alpha=0.3, label='Error')
    ax.legend(bbox_to_anchor = (0.05, 0.9), loc = 'upper left', borderaxespad = 0., fontsize=10)
    ax1 = ax.twinx()
    #ax1.plot(np.linspace(0, 800, time_scale), np.log10(simulator_ols.ds_hist[:time_scale])+2, label=r'$\log(F(\widehat u(t)))$', color='r', linewidth=.5, linestyle='dashed')
    #ax1.plot(simulator_ols.ds_hist[1:time_scale], color='r', linewidth=.5, linestyle='dashed')
    #ax.plot(simulator_ablation.ds_hist[1:time_scale], color='b', linewidth=.5, linestyle='dashed')
    #ax1.legend(bbox_to_anchor = (0.1, 0.6), loc = 'upper left', borderaxespad = 0., fontsize=10)
    ax1.set_ylabel(r'$\log(F(\widehat u(t)))$', fontsize=10)
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax1.yaxis.set_tick_params(labelsize=labelsize)
    ax.set_ylabel(r'$\| u(t) - \widehat u(t) \|_{2}^2/\| u(t) \|_{2}^2$', fontsize=10)
    plt.savefig('../../fig/exp1/{}/2.svg'.format(simu_type), dpi = 1000, bbox_inches='tight', pad_inches=0, format='svg')
    plt.savefig('../../fig/exp1/{}/2.pdf'.format(simu_type), dpi = 1000, bbox_inches='tight', pad_inches=0)
    plt.show()'''

    # For the comparison bewteen RD & NS, we need to show following features of our figures
    # 1. Two figures should be of the same time scope: same \Delta t & same number of step_num
    # 2. RD should suffer from less severe distribution shift comparing to NS, meaning that 
    # both the error and ds should be smaller than NS.
    # 3. Currently, the ds legend in NS is too small.

if __name__ == '__main__':
    main()