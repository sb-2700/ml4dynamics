# This is the script for the figure \cref{NS-cmp}, we only use the `else` block 
# to do the plot
from utils import *
from simulator import *
from model import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


u, v, label = read_data('NS/nx128Re400.npz')
type = 'NS'
if type == 'RD':
    nx = 64
    ny = 64
    step_num = 100
    dt = 0.01
else:
    nx = 128
    ny = 32
    step_num = 2000
    dt = 0.01


# OSL model
model_ols = torch.load('model/ns_modelRe400.pt')
# TR model
#model_tr = torch.load('model/ns_modelRe400r.pt')
# auto-encoder
model_ed = torch.load('model/ed_modelRe400.pt')


u64_np = copy.deepcopy(u.numpy())
v64_np = copy.deepcopy(v.numpy())
print(u64_np.shape)
u64_np = u64_np.reshape([step_num, nx+2, ny+2])
v64_np = v64_np.reshape([step_num, nx+2, ny+1])

 
simulator_ols = NS_Simulator(model_ols, model_ed, device, file_path='../fig/exp5',u_hist=u64_np, v_hist=v64_np, step_num=2000,dt=0.01)
#simulator_tr = NS_Simulator(model_tr, ed_model, device, file_path='../fig/exp5',u_hist=u64_np, v_hist=v64_np, step_num=2000,dt=0.01)
simulator_ols.simulator() 
#simulator_tr.simulator() 


plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi']  = 300 
fig = plt.figure()
blowup_time = 120
simu_time = 200
plot_case = False


if plot_case:
    ax2 = fig.add_subplot(221)
    ax2.plot(simulator_tr.error_hist[:240], label='TR', color='b', linewidth=.5)
    ax2.plot(simulator_ols.error_hist[:blowup_time], label='OLS', color='r', linewidth=.5)
    #ax2.set_title('error comparison')
    ax2.legend(bbox_to_anchor = (0.6, 0.9), loc = 'upper left', borderaxespad = 0., fontsize=5)
    ax2.xaxis.set_tick_params(labelsize=5)
    ax2.yaxis.set_tick_params(labelsize=5)
    #ax2.set_xlabel('time step', fontsize=5)
    ax2.set_ylabel('error', fontsize=5)


    ax1 = fig.add_subplot(222)
    ax1.plot(np.log10(simulator_tr.ds_hist[:240]), label='TR', color='b', linewidth=.5)
    ax1.plot(np.log10(simulator_ols.ds_hist[:blowup_time]), label='OLS', color='r', linewidth=.5)
    #ax1.set_title('distribution shift comparison')
    ax1.legend(bbox_to_anchor = (0.6, 0.9), loc = 'upper left', borderaxespad = 0., fontsize=5)
    ax1.xaxis.set_tick_params(labelsize=5)
    ax1.yaxis.set_tick_params(labelsize=5)
    #ax1.set_xlabel('time step', fontsize=5)
    ax1.set_ylabel('distribution shift', fontsize=5)


    ax3 = fig.add_subplot(223)
    ax3.plot(simulator_ols.error_hist[:blowup_time], label='error', color='b', linewidth=.5)
    ax31 = ax3.twinx()
    ax31.plot(simulator_ols.ds_hist[:blowup_time], label='distribution shift', color='r', linewidth=.5)
    ax3.legend(bbox_to_anchor = (0.4, 0.9), loc = 'upper left', borderaxespad = 0., fontsize=5)
    ax31.legend(bbox_to_anchor = (0.4, 0.8), loc = 'upper left', borderaxespad = 0., fontsize=5)
    ax3.xaxis.set_tick_params(labelsize=5)
    ax3.yaxis.set_tick_params(labelsize=5)
    ax31.yaxis.set_tick_params(labelsize=5)
    ax3.set_xlabel('time step', fontsize=5)
    ax3.set_ylabel('error', fontsize=5)


    ax4 = fig.add_subplot(224)
    ax4.plot(simulator_tr.error_hist[:240], label='error', color='b', linewidth=.5)
    ax41 = ax4.twinx()
    ax41.plot(simulator_tr.ds_hist[:240], label='distribution shift', color='r', linewidth=.5)
    ax4.legend(bbox_to_anchor = (0.1, 0.9), loc = 'upper left', borderaxespad = 0., fontsize=5)
    ax41.legend(bbox_to_anchor = (0.1, 0.8), loc = 'upper left', borderaxespad = 0., fontsize=5)
    ax4.xaxis.set_tick_params(labelsize=5)
    ax4.yaxis.set_tick_params(labelsize=5)
    ax41.yaxis.set_tick_params(labelsize=5)
    ax4.set_xlabel('time step', fontsize=5)
    ax41.set_ylabel('distribution shift', fontsize=5)
    plt.savefig('fig/NSRe100.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    #plt.plot(ns_simulator.ds_hist[:120])


else:
    ax = fig.add_subplot(111)
    ax.plot(simulator_tr.error_hist[:simu_time], label='TR error', color='b', linewidth=.5)
    ax.plot(simulator_ols.error_hist[:blowup_time], label='OLS error', color='r', linewidth=.5)
    #ax2.set_title('error comparison')
    ax.legend(bbox_to_anchor = (0.6, 0.9), loc = 'upper left', borderaxespad = 0., fontsize=5)
    ax.xaxis.set_tick_params(labelsize=5)
    ax.yaxis.set_tick_params(labelsize=5)
    #ax2.set_xlabel('time step', fontsize=5)
    ax.set_ylabel('error', fontsize=5)
    ax1 = ax.twinx()
    ax1.plot(np.log10(simulator_tr.ds_hist[:simu_time]), label='TR ds', color='b', linewidth=.5, linestyle='dashed')
    ax1.plot(np.log10(simulator_ols.ds_hist[:blowup_time]), label='OLS ds', color='r', linewidth=.5, linestyle='dashed')
    ax1.legend(bbox_to_anchor = (0.6, 0.9), loc = 'upper left', borderaxespad = 0., fontsize=5)
    ax1.set_ylabel('distribution shift', fontsize=5)