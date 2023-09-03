# This script plot \cref{RD-beta-ds}
from utils import *
from simulator import *
from models import *

r.seed(0)
n, beta, Re, type, GPU, ds_parameter = parsing()
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

if type == 'RD':
    ds_parameter = [2, 4, 6, 8, 10]
    ds_parameter_ = [10, 6, 8, 4, 2]
    #ds_parameter_ = ds_parameter
    t_array = [2, 30, 60, 90]
    t_array_ = [160, 320, 480, 800]
else:
    ds_parameter = [100, 200, 300, 400, 500]
    t_array = [0, 300, 600, 900]

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi']  = 300 
labelsize = 10
fontsize = 10
fig = plt.figure()
ax = fig.add_subplot(111)
ax1 = ax.twinx()
c = ['b', 'r', 'g', 'y']
for i in range(5):
    
    arg, u, v, label = read_data('../../data/{}/{}-{}.npz'.format(type, n, ds_parameter[i]))

    nx, ny, dt, T, label_dim, traj_num, step_num, test_index, u, v, label = preprocessing(arg, type, u, v, label, device, flag=False)


    # loading models
    if type == 'NS':
        model_ols = UNet([2,4,8,16,32,64,1]).to(device)
        u64_np = copy.deepcopy(u[test_index].numpy()).reshape([step_num, nx+2, ny+2])
        v64_np = copy.deepcopy(v[test_index].numpy()).reshape([step_num, nx+2, ny+1])
    else:
        model_ols = UNet().to(device)
        u64_np = copy.deepcopy(u[test_index].numpy()).reshape([step_num, nx, ny])
        v64_np = copy.deepcopy(v[test_index].numpy()).reshape([step_num, nx, ny])
    model_ols.load_state_dict(torch.load('../../models/{}/OLS-{}-{}.pth'.format(type, n, ds_parameter[i]), 
                                    map_location=torch.device('cpu')))
    model_ols.eval()
    model_ed = EDNet().to(device)
    model_ed.load_state_dict(torch.load('../../models/{}/ED-{}-{}.pth'.format(type, n, ds_parameter[i]),
                                    map_location=torch.device('cpu')))
    model_ed.eval()


    if type == 'RD':
        simulator_ols = RD_Simulator(model_ols, 
                                    model_ed, 
                                    device, 
                                    u_hist=u64_np, 
                                    v_hist=v64_np, 
                                    step_num=step_num,
                                    dt=dt)
    else:
        simulator_ols = NS_Simulator(model_ols, 
                                    model_ed, 
                                    device, 
                                    u_hist=u64_np, 
                                    v_hist=v64_np, 
                                    step_num=step_num,
                                    dt=dt)
    simulator_ols.simulator()  

    for j in range(4):
        if i == 0:
            ax.scatter(ds_parameter_[i]/10, simulator_ols.error_hist[t_array[j]], c=c[j], marker='o', label = r'$t={}$'.format(t_array_[j]))
        else:
            ax.scatter(ds_parameter_[i]/10, simulator_ols.error_hist[t_array[j]], c=c[j], marker='o')
        ax1.scatter(ds_parameter_[i]/10, np.log10(simulator_ols.ds_hist[t_array[j]]), c=c[j], marker='x')
    #ax.plot(simulator_ols.error_hist[:-10], label='OLS error, beta={:1f}'.format(beta[i]), color=c[i], linewidth=.5)
    #ax.legend(bbox_to_anchor = (0.05, 0.95), loc = 'upper left', borderaxespad = 0., fontsize=fontsize)
    #ax1.plot(np.log10(simulator_ols.ds_hist[:-10]*np.linspace(0.01, 1, 90)), label='OLS log(ds), beta='+str(beta), color=c[i], linewidth=.5, linestyle='dashed')
    #ax1.plot(np.log10(simulator_ols.ds_hist[:-10]), label='OLS log(ds), beta={:1f}'.format(beta[i]), color=c[i], linewidth=.5, linestyle='dashed')
    #ax1.legend(bbox_to_anchor = (0.05, 0.8), loc = 'upper left', borderaxespad = 0., fontsize=fontsize)


ax.xaxis.set_tick_params(labelsize=labelsize)
ax.yaxis.set_tick_params(labelsize=labelsize)
ax1.yaxis.set_tick_params(labelsize=labelsize)
ax.set_xlabel(r'$\beta$', fontsize=fontsize)
ax.set_ylabel(r'$\left\| u(t) - \widehat u(t) \right\|_{2}^2$', fontsize=fontsize)
ax1.set_ylabel(r'$\log(F(\widehat u(t)))$', fontsize=fontsize)
ax.legend()
plt.savefig('../../fig/exp3/{}/RDbeta-ds.pdf'.format(type))
plt.show()