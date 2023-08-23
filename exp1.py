# This script plot \cref{RD-ds}
# This script plot \cref{NS-ds}
import argparse
import numpy.random as r
from utils import *
from simulator import *
from model import *
from matplotlib import pyplot as plt
from matplotlib import cm


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--Re', type=int, default=1)
parser.add_argument('--n', type=int, default=64)
parser.add_argument('--type', type=str, default='RD')
args = parser.parse_args()
beta = args.beta
Re = args.Re
n = args.n
type = args.type


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
r.seed(0)


type = 'RD'
if type == 'RD':
    u, v, label = read_data('RD/n{}beta{:.1f}.npz'.format(n, beta))
    dt = 0.01
    traj_num = u.shape[0]
    step_num = u.shape[1]
    nx = n
    ny = n
elif type == 'NS':
    u, v, label = read_data('NS/n{}Re{:.1f}.npz'.format(n, beta))
    dt = 0.01
    dt = 0.01
    traj_num = u.shape[0]
    step_num = u.shape[1]
    nx = 130
    ny = 34
else:
    print('Unknown fluid type!!!')
    raise NameError
    


# loading models
model_ols = UNet().to(device)
model_ols.load_state_dict(torch.load('model/RD/OLS-n{}beta{:.1f}.pth'.format(n, beta)))
model_ols.eval()
model_ed = EDNet().to(device)
model_ed.load_state_dict(torch.load('model/RD/ED-n{}beta{:.1f}.pth'.format(n, beta)))
model_ed.eval()


test_index = 0
test_index = int(np.floor(r.rand() * traj_num))     # setting the same seed means we will have the same test traj
u64_np = copy.deepcopy(u[test_index].numpy()).reshape([step_num, nx, ny])
v64_np = copy.deepcopy(v[test_index].numpy()).reshape([step_num, nx, ny])


if type == 'RD':
    simulator_ols = RD_Simulator(model_ols, model_ed, device, u_hist=u64_np, v_hist=v64_np, step_num=step_num,dt=dt)
else:
    simulator_ols = NS_Simulator(model_ols, model_ed, device, u_hist=u64_np, v_hist=v64_np, step_num=step_num,dt=dt)
simulator_ols.simulator()  


plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi']  = 300 
width = 10
height = 4
#################################################################
#               Plot of the simulation result                   #
#################################################################
fig = plt.figure(figsize=(width, height))
fig_num = 5
hspace = 0.01
wspace = -0.35
fraction = 0.024
pad = 0.001
if type == 'RD':
    t_array = np.array([0, 20, 40, 60, 80])
ax1 = []
ax2 = []
for i in range(fig_num):
    ax1.append(fig.add_subplot(2,fig_num,i+1))
    ax2.append(fig.add_subplot(2,fig_num,i+fig_num+1))
fig.tight_layout()
for i in range(fig_num):
    ax1[i].imshow(simulator_ols.u_hist[t_array[i]], cmap=cm.jet)
    ax1[i].set_axis_off()
    im = ax2[i].imshow(simulator_ols.u_hist_simu[t_array[i]], cmap=cm.jet)
    ax2[i].set_axis_off()
plt.subplots_adjust(hspace=hspace)
plt.subplots_adjust(wspace=wspace)
for i in range(fig_num):
    cbar = fig.colorbar(im, ax=[ax1[i], ax2[i]], fraction=fraction, pad=pad, orientation='horizontal')
    cbar.ax.tick_params(labelsize=5)
plt.savefig('../fig/exp1/{}/1.jpg'.format(type), dpi = 1000, bbox_inches='tight', pad_inches=0)
#plt.show()
plt.clf()


#################################################################
#              Plot of the ds & error comparison                #
#################################################################
fig = plt.figure(figsize=(width//2, height))
ax = fig.add_subplot(111)
ax.plot(simulator_ols.error_hist, label='OLS error', color='r', linewidth=.5)
ax.legend(bbox_to_anchor = (0.1, 0.9), loc = 'upper left', borderaxespad = 0., fontsize=5)
ax.xaxis.set_tick_params(labelsize=5)
ax.yaxis.set_tick_params(labelsize=5)
ax.set_ylabel('error', fontsize=5)
ax1 = ax.twinx()
ax1.plot(np.log10(simulator_ols.ds_hist[:-1]), label='OLS log(ds)', color='r', linewidth=.5, linestyle='dashed')
ax1.legend(bbox_to_anchor = (0.1, 0.8), loc = 'upper left', borderaxespad = 0., fontsize=5)
ax1.set_ylabel('log of dds', fontsize=5)
plt.savefig('../fig/exp1/{}/2.jpg'.format(type), dpi = 1000, bbox_inches='tight', pad_inches=0)
#plt.show()