# This is the script for the figure \cref{NS-cmp}, we only use the `else` block 
# to do the plot
from pathlib import Path
import sys
ROOT_PATH = str(Path(__file__).parent.parent)
sys.path.append(ROOT_PATH)

from src.utils import *
from src.simulator import *
from src.models import *
from test import test_simulator

def main():
    simulator_list_ols = []
    simulator_list_tr = []
    simu_type = 'RD'
    n = 128

    for i in range(10):
        simulator_list_ols.append(test_simulator(n=n, test_index=i, simu_type='RD', model_type='OLS')[0])
        simulator_list_tr.append(test_simulator(n=n, test_index=i, simu_type='RD', model_type='TR')[0])
    simu_type = test_simulator(test_index=i, model_type='TR')[1]

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi']  = 300 
    width = 10
    height = 4
    labelsize = 10
    #################################################################
    #               Plot of the simulation result                   #
    # we only plot the result of the test history                   #
    #################################################################
    test_index = 0
    simulator_ols = simulator_list_ols[test_index]
    simulator_tr = simulator_list_tr[test_index]

    fig = plt.figure(figsize=(width, height))
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
    fig_num = 5
    hspace = 0.01
    wspace = -0.765
    fraction = 0.016
    pad = 0.001
    time_scale = 500
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
        im_ = ax1[i].imshow(simulator_ols.u_hist[t_array[i]][x_left:x_right, y_left:y_right], cmap=cm.jet)
        #cbar = fig.colorbar(im, ax=ax1[i], fraction=fraction, pad=pad, orientation='horizontal')
        #cbar.ax.tick_params(labelsize=5)
        ax1[i].set_axis_off()
        im.append(im_)
        im_ = ax2[i].imshow(simulator_ols.u_hist_simu[t_array[i]][x_left:x_right, y_left:y_right], cmap=cm.jet)
        #cbar = fig.colorbar(im, ax=ax2[i], fraction=fraction, pad=pad, orientation='horizontal')
        #cbar.ax.tick_params(labelsize=5)
        ax2[i].set_axis_off()
        im_ = ax3[i].imshow(simulator_tr.u_hist_simu[t_array[i]][x_left:x_right, y_left:y_right], cmap=cm.jet)
        #cbar = fig.colorbar(im, ax=ax3[i], fraction=fraction, pad=pad, orientation='horizontal')
        #cbar.ax.tick_params(labelsize=5)
        ax3[i].set_axis_off()
        
    plt.subplots_adjust(hspace=hspace)
    plt.subplots_adjust(wspace=wspace)
    for i in range(fig_num):
        cbar = fig.colorbar(im[i], ax=[ax1[i], ax2[i], ax3[i]], fraction=fraction, pad=pad, orientation='horizontal')
        cbar.ax.tick_params(labelsize=5)
    plt.savefig('../../fig/exp4/{}/1.pdf'.format(simu_type), dpi = 1000, bbox_inches='tight', pad_inches=0)
    plt.savefig('../../fig/exp4/{}/1.svg'.format(simu_type), dpi = 1000, bbox_inches='tight', pad_inches=0, format='svg')
    plt.show()
    #plt.clf()

    #################################################################
    #              Plot of the ds & error comparison                #
    #################################################################
    '''avg_err_hist_ols = 0
    avg_err_hist_tr = 0
    avg_ds_hist_ols = 0
    avg_ds_hist_tr = 0
    std_err_hist_ols = 0
    std_err_hist_tr = 0
    std_ds_hist_ols = 0
    std_ds_hist_tr = 0
    for i in range(1):
        avg_err_hist_ols = avg_err_hist_ols + simulator_list_ols[i].error_hist
        avg_err_hist_tr = avg_err_hist_tr + simulator_list_tr[i].error_hist
        avg_ds_hist_ols = avg_ds_hist_ols + simulator_list_ols[i].ds_hist
        avg_ds_hist_tr = avg_ds_hist_tr + simulator_list_tr[i].ds_hist
    avg_err_hist_tr = avg_err_hist_tr/10
    avg_err_hist_ols = avg_err_hist_ols/10
    for i in range(1):
        std_err_hist_ols = std_err_hist_ols + (simulator_list_ols[i].error_hist - avg_err_hist_ols)**2
        std_err_hist_tr = std_err_hist_tr + (simulator_list_tr[i].error_hist - avg_err_hist_tr)**2
        std_ds_hist_ols = std_ds_hist_ols + (simulator_list_ols[i].ds_hist - avg_ds_hist_ols)**2
        std_ds_hist_tr = std_ds_hist_tr + (simulator_list_tr[i].ds_hist - avg_ds_hist_tr)**2
    std_err_hist_ols = np.sqrt(std_err_hist_ols/10)
    std_err_hist_tr = np.sqrt(std_err_hist_tr/10)
    std_ds_hist_ols = np.sqrt(std_ds_hist_ols/10)
    std_ds_hist_tr = np.sqrt(std_ds_hist_tr/10)

    fig = plt.figure(figsize=(width, height//2))
    ax = fig.add_subplot(111)
    #time_scale = avg_ds_hist_ols.shape[0]
    time_scale = 500
    plot_interval = 25
    #pdb.set_trace()
    ax.errorbar(np.linspace(0, time_scale, time_scale//plot_interval), 
                avg_err_hist_ols[:time_scale:plot_interval], 
                std_err_hist_ols[:time_scale:plot_interval], 
                fmt='.', label=r'$\left\| \widehat u_{OLS} - u \right\|_{2}^2$', ms=0.1)
    ax.errorbar(np.linspace(0, time_scale, time_scale//plot_interval), 
                avg_err_hist_tr[:time_scale:plot_interval], 
                std_err_hist_tr[:time_scale:plot_interval], 
                fmt='o', label=r'$\left\| \widehat u_{TR} - u \right\|_{2}^2$', ms=0.1)
    #ax.plot(np.linspace(0, 800, time_scale), simulator_ols.error_hist[:time_scale], label='OLS error', color='r', linewidth=.5)
    ax.legend(bbox_to_anchor = (0.1, 0.9), loc = 'upper left', borderaxespad = 0., fontsize=10)
    ax1 = ax.twinx()
    ax.errorbar(np.linspace(0, time_scale, time_scale//plot_interval), 
                avg_ds_hist_ols[:time_scale:plot_interval], 
                std_ds_hist_ols[:time_scale:plot_interval], 
                fmt='.', ms=0.1)
    ax.errorbar(np.linspace(0, time_scale, time_scale//plot_interval), 
                avg_ds_hist_tr[:time_scale:plot_interval], 
                std_ds_hist_tr[:time_scale:plot_interval], 
                fmt='o', ms=0.1)
    ax1.legend(bbox_to_anchor = (0.1, 0.7), loc = 'upper left', borderaxespad = 0., fontsize=10)
    ax1.set_ylabel(r'$\log(F(\widehat u(t)))$', fontsize=10)
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax1.yaxis.set_tick_params(labelsize=labelsize)
    ax.set_ylabel(r'$\left\| u(t) - \widehat u(t) \right\|_{F}^2$', fontsize=10)
    plt.savefig('../../fig/exp4/{}/2.svg'.format(simu_type), dpi = 1000, bbox_inches='tight', pad_inches=0, format='svg')
    plt.show()'''

if __name__ == '__main__':
    main()