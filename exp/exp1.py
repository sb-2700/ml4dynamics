# This script plot \cref{RD-ds}
# This script plot \cref{NS-ds}
from utils import *
from simulator import *
from models import *
from test import test_simulator

r.seed(0)
test_index = int(np.floor(r.rand() * 10))
simulator_ols, type = test_simulator()

plt.rcParams['savefig.dpi'] = 500
plt.rcParams['figure.dpi']  = 500 
width = 7.5
height = 3
labelsize = 10
#################################################################
#               Plot of the simulation result                   #
#################################################################
fig = plt.figure(figsize=(width, height))
fig_num = 5
hspace = 0.01
wspace = -0.40
fraction = 0.024
pad = 0.001
time_scale = 140
t_array = (np.array([0.0, 0.2, 0.4, 0.6, 1.0])*time_scale).astype(int)
print(t_array)
ax1 = []
ax2 = []
im = []
for i in range(fig_num):
    ax1.append(fig.add_subplot(2,fig_num,i+1))
    ax2.append(fig.add_subplot(2,fig_num,i+fig_num+1))
fig.tight_layout()
for i in range(fig_num):
    ax1[i].imshow(simulator_ols.u_hist[t_array[i]][1:33, 1:-1], cmap=cm.jet)
    ax1[i].set_axis_off()
    im.append(ax2[i].imshow(simulator_ols.u_hist_simu[t_array[i]][1:33, 1:-1], cmap=cm.jet))
    ax2[i].set_axis_off()
plt.subplots_adjust(hspace=hspace)
plt.subplots_adjust(wspace=wspace)
for i in range(fig_num):
    cbar = fig.colorbar(im[i], ax=[ax1[i], ax2[i]], fraction=fraction, pad=pad, orientation='horizontal')
    cbar.ax.tick_params(labelsize=5)
#plt.savefig('../../fig/exp1/{}/1.svg'.format(type), dpi = 1000, bbox_inches='tight', pad_inches=0, format='svg')
plt.savefig('../../fig/exp1/{}/1.pdf'.format(type), dpi = 1000, bbox_inches='tight', pad_inches=0)
#plt.show()
plt.clf()


#################################################################
#              Plot of the ds & error comparison                #
#################################################################
fig = plt.figure(figsize=(width, height/2))
ax = fig.add_subplot(111)
ax.plot(np.linspace(0, 800, time_scale), simulator_ols.error_hist[:time_scale]*10, label=r'$\left\| u(t) - \widehat u(t) \right\|_{2}^2$', color='r', linewidth=.5)
ax.legend(bbox_to_anchor = (0.1, 0.9), loc = 'upper left', borderaxespad = 0., fontsize=10)
ax1 = ax.twinx()
ax1.plot(np.linspace(0, 800, time_scale), np.log10(simulator_ols.ds_hist[:time_scale])+2, label=r'$\log(F(\widehat u(t)))$', color='r', linewidth=.5, linestyle='dashed')
ax1.legend(bbox_to_anchor = (0.1, 0.6), loc = 'upper left', borderaxespad = 0., fontsize=10)
ax1.set_ylabel(r'$\log(F(\widehat u(t)))$', fontsize=10)
ax.xaxis.set_tick_params(labelsize=labelsize)
ax1.yaxis.set_tick_params(labelsize=labelsize)
ax.set_ylabel(r'$\left\| u(t) - \widehat u(t) \right\|_{2}^2$', fontsize=10)
#plt.savefig('../../fig/exp1/{}/2.svg'.format(type), dpi = 1000, bbox_inches='tight', pad_inches=0, format='svg')
plt.savefig('../../fig/exp1/{}/2.pdf'.format(type), dpi = 1000, bbox_inches='tight', pad_inches=0)
plt.show()


# For the comparison bewteen RD & NS, we need to show following features of our figures
# 1. Two figures should be of the same time scope: same \Delta t & same number of step_num
# 2. RD should suffer from less severe distribution shift comparing to NS, meaning that 
# both the error and ds should be smaller than NS.
# 3. Currently, the ds legend in NS is too small.