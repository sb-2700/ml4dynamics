from models import *
from simulator import *
from utils import *


def test_simulator(n=64, model_type='aOLS', simu_type='RD', ds_parameter=1, test_index=0, true_model=0):
    r.seed(0)
    device = torch.device('cpu')

    arg, U, label = read_data('../../data/{}/{}-{}.npz'.format(simu_type, n, ds_parameter))

    nx, ny, dt, T, label_dim, traj_num, step_num, U, label = preprocessing(arg, simu_type, U, label, device, flag=False)

    #print('Testing {} model with n = {}, ds_parameter = {} on {}-th trajectory'.format(simu_type, n, ds_parameter, test_index))
        
    if simu_type == 'NS':
        model = UNet([2,4,8,32,64,128,1]).to(device)
        u_np = copy.deepcopy(U[test_index,:,0].numpy()).reshape([step_num, nx+2, ny+2])
        v_np = copy.deepcopy(U[test_index,:,1,:,1:].numpy()).reshape([step_num, nx+2, ny+1])
    else:
        model = UNet([2,4,8,32,64,128,2]).to(device)
        u_np = copy.deepcopy(U[test_index,:,0].numpy()).reshape([step_num, nx, ny])
        v_np = copy.deepcopy(U[test_index,:,1].numpy()).reshape([step_num, nx, ny])
    if model_type == 'true':
        model = true_model
    else:
        model.load_state_dict(torch.load('../../models/{}/{}-{}-{}.pth'.format(simu_type, model_type, n, ds_parameter), 
                                            map_location=torch.device('cpu')))
        model.eval()
    model_ed = EDNet(channel_array=[2,4,8,16,32,64]).to(device)
    model_ed.load_state_dict(torch.load('../../models/{}/ED-{}-{}.pth'.format(simu_type, n, ds_parameter),
                                        map_location=torch.device('cpu')))
    model_ed.eval()

    if simu_type == 'RD':
        simulator = RD_Simulator(model, 
                                    model_ed, 
                                    device, 
                                    u_hist=u_np, 
                                    v_hist=v_np, 
                                    step_num=step_num,
                                    dt=dt,
                                    n=n,
                                    gamma=ds_parameter/20)
    else:
        simulator = NS_Simulator(model, 
                                    model_ed, 
                                    device, 
                                    u_hist=u_np, 
                                    v_hist=v_np, 
                                    step_num=step_num,
                                    dt=dt,
                                    nx=nx,
                                    ny=ny,
                                    Re=ds_parameter)
    '''simulator.ablation_study = True
    simulator.simulator() 
    simulator.ds_hist_ablation = copy.deepcopy(simulator.ds_hist)
    simulator.error_hist_ablation = copy.deepcopy(simulator.error_hist)'''
    simulator.ablation_study = False
    simulator.simulator()  
    return simulator, simu_type