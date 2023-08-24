# This script plot \cref{RD-beta-ds}
import argparse
from utils import *
from simulator import *
from model import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi']  = 300 
labelsize = 7
fontsize = 5
fig = plt.figure()
ax = fig.add_subplot(111)
ax1 = ax.twinx()
c = ['b', 'r', 'b', 'g', 'y']
for i in range(1, 4):
    beta = i/10
    u, v, label = read_data('RD/n64beta'+str(beta)+'.npz')
    type = 'RD'
    if type == 'RD':
        nx = 64
        ny = 64
        step_num = 100
        dt = 0.01
    else:
        nx = 130
        ny = 34
        step_num = 2000
        dt = 0.01


    # loading models
    model_ols = UNet().to(device)
    model_ols.load_state_dict(torch.load('model/RD/OLS-n64beta'+str(beta)+'e-1.pth'))
    model_ols.eval()
    model_ed = UNet().to(device)
    model_ed.load_state_dict(torch.load('model/RD/ED-n64beta'+str(beta)+'e-1.pth'))
    model_ed.eval()


    u64_np = copy.deepcopy(u.numpy()).reshape([step_num, nx, ny])
    v64_np = copy.deepcopy(v.numpy()).reshape([step_num, nx, ny])


    if type == 'RD':
        simulator_ols = RD_Simulator(model_ols, model_ed, device, file_path='../fig/exp1', u_hist=u64_np, v_hist=v64_np, step_num=step_num,dt=dt)
    else:
        simulator_ols = NS_Simulator(model_ols, model_ed, device, file_path='../fig/exp1', u_hist=u64_np, v_hist=v64_np, step_num=step_num,dt=dt)
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