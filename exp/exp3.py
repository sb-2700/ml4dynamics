# This script plot \cref{RD-beta-ds}
import argparse
from utils import *
from simulator import *
from models import *
from matplotlib import pyplot as plt
from matplotlib import cm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
r.seed(0)

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi']  = 300 
labelsize = 7
fontsize = 5
n = 64
fig = plt.figure()
ax = fig.add_subplot(111)
ax1 = ax.twinx()
c = ['b', 'r', 'b', 'g', 'y']
beta = [0.2, 0.4, 0.6, 0.8, 1.0]
Re = [100, 200, 300, 400, 500]
type = 'RD'
Re = 100
for i in range(5):
    ds_parameter = int(beta[i]*10)
    arg, u, v, label = read_data('../../data/{}/{}-{}.npz'.format(type, n, ds_parameter))
    nx, ny, dt, T, label_dim, traj_num, step_num, test_index, u, v, label = preprocessing(arg, type, u, v, label, device, flag=False)

    print('Plotting {} model with n = {}, beta = {} ...'.format(type, n, ds_parameter))


    # loading models
    if type == 'NS':
        model_ols = UNet([2,4,8,16,32,64,1]).to(device)
        u64_np = copy.deepcopy(u[test_index].numpy()).reshape([step_num, nx+2, ny+2])
        v64_np = copy.deepcopy(v[test_index].numpy()).reshape([step_num, nx+2, ny+1])
    else:
        model_ols = UNet().to(device)
        u64_np = copy.deepcopy(u[test_index].numpy()).reshape([step_num, nx, ny])
        v64_np = copy.deepcopy(v[test_index].numpy()).reshape([step_num, nx, ny])
    model_ols.load_state_dict(torch.load('../../models/{}/OLS-{}-{}.pth'.format(type, n, ds_parameter), 
                                    map_location=torch.device('cpu')))
    model_ols.eval()
    model_ed = EDNet().to(device)
    model_ed.load_state_dict(torch.load('../../models/{}/ED-{}-{}.pth'.format(type, n, ds_parameter),
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


    ax.plot(simulator_ols.error_hist[:-10], label='OLS error, beta='+str(beta), color=c[i], linewidth=.5)
    ax.legend(bbox_to_anchor = (0.05, 0.95), loc = 'upper left', borderaxespad = 0., fontsize=fontsize)
    ax1.plot(np.log10(simulator_ols.ds_hist[:-10]*np.linspace(0.01, 1, 90)), label='OLS log(ds), beta='+str(beta), color=c[i], linewidth=.5, linestyle='dashed')
    ax1.legend(bbox_to_anchor = (0.05, 0.8), loc = 'upper left', borderaxespad = 0., fontsize=fontsize)


ax.xaxis.set_tick_params(labelsize=labelsize)
ax.yaxis.set_tick_params(labelsize=labelsize)
ax1.yaxis.set_tick_params(labelsize=labelsize)
ax.set_ylabel('error', fontsize=labelsize)
ax1.set_ylabel('log of dds', fontsize=labelsize)
plt.savefig('../fig/RDbeta-ds.jpg')
plt.show()