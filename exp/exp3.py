# This script plot \cref{RD-beta-ds}
from utils import *
from simulator import *
from models import *

def main():
    r.seed(0)
    device = torch.device('cpu')
    type = 'NS'
    n = 128

    if type == 'RD':
        ds_parameter = [1,2,3,4,5]
        ds_parameter_ = ds_parameter
        #ds_parameter_ = ds_parameter
        t_array = [2, 30, 60, 90]
        t_array_ = [160, 320, 480, 800]
    else:
        ds_parameter = [100, 200, 300, 400, 500]
        ds_parameter_ = [100, 200, 300, 400, 500]
        t_array = [0, 300, 600, 900]
        t_array_ = [0, 300, 600, 900]

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi']  = 300 
    labelsize = 10
    fontsize = 10
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = ax.twinx()
    c = ['b', 'r', 'g', 'y']
    test_index = 0
    for i in range(5):
        
        arg, U, label = read_data('../../data/{}/{}-{}.npz'.format(type, n, ds_parameter[i]))

        nx, ny, dt, T, label_dim, traj_num, step_num, U, label = preprocessing(arg, type, U, label, device, flag=False)


        # loading models
        if type == 'NS':
            model_ols = UNet([2,4,8,32,64,128,1]).to(device)
            u64_np = copy.deepcopy(U[test_index,:,0].numpy()).reshape([step_num, nx+2, ny+2])
            v64_np = copy.deepcopy(U[test_index,:,1,:,1:].numpy()).reshape([step_num, nx+2, ny+1])
        else:
            model_ols = UNet([2,4,8,32,64,128,2]).to(device)
            u64_np = copy.deepcopy(U[test_index,:,0].numpy()).reshape([step_num, nx, ny])
            v64_np = copy.deepcopy(U[test_index,:,1].numpy()).reshape([step_num, nx, ny])
        model_ols.load_state_dict(torch.load('../../models/{}/OLS-{}-{}.pth'.format(type, n, ds_parameter[i]), 
                                        map_location=torch.device('cpu')))
        model_ols.eval()
        model_ed = EDNet(channel_array=[2,4,8,16,32,64]).to(device)
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
        simulator_ols.ablation_study = False
        simulator_ols.simulator()  

        for j in range(4):
            if i == 0:
                ax.scatter(ds_parameter_[i], simulator_ols.error_hist[t_array[j]], c=c[j], marker='o', label = r'$t={}$'.format(t_array_[j]))
            else:
                ax.scatter(ds_parameter_[i], simulator_ols.error_hist[t_array[j]], c=c[j], marker='o')
            ax1.scatter(ds_parameter_[i], np.log10(simulator_ols.ds_hist[t_array[j]]), c=c[j], marker='x')
        #ax.plot(simulator_ols.error_hist[:-10], label='OLS error, beta={:1f}'.format(beta[i]), color=c[i], linewidth=.5)
        #ax.legend(bbox_to_anchor = (0.05, 0.95), loc = 'upper left', borderaxespad = 0., fontsize=fontsize)
        #ax1.plot(np.log10(simulator_ols.ds_hist[:-10]*np.linspace(0.01, 1, 90)), label='OLS log(ds), beta='+str(beta), color=c[i], linewidth=.5, linestyle='dashed')
        #ax1.plot(np.log10(simulator_ols.ds_hist[:-10]), label='OLS log(ds), beta={:1f}'.format(beta[i]), color=c[i], linewidth=.5, linestyle='dashed')
        #ax1.legend(bbox_to_anchor = (0.05, 0.8), loc = 'upper left', borderaxespad = 0., fontsize=fontsize)


    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    ax1.yaxis.set_tick_params(labelsize=labelsize)
    ax.set_xlabel('Re', fontsize=fontsize)
    ax.set_ylabel(r'$\| u(t) - \widehat u(t) \|_{2}^2$', fontsize=fontsize)
    ax1.set_ylabel(r'$\log(F(\widehat u(t)))$', fontsize=fontsize)
    ax.legend()
    plt.savefig('../../fig/exp3/{}/{}Re-ds.pdf'.format(type, type), dpi = 1000, bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == '__main__':
    main()