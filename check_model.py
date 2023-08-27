import numpy as np
import numpy.random as r
import torch as torch
from utils import *
from models import *
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--Re', type=int, default=400)
parser.add_argument('--n', type=int, default=64)
parser.add_argument('--type', type=str, default='NS')
parser.add_argument('--GPU', type=int, default=0)
parser.add_argument('--i', type=int, default=5)
parser.add_argument('--end', type=int, default=500)
args = parser.parse_args()
beta = args.beta
Re = args.Re
n = args.n
type = args.type
GPU = args.GPU
if type == 'RD':
    ds_parameter = int(beta*10)
else:
    ds_parameter = Re
i = args.i
end = args.end
r.seed(0)
device = torch.device('cpu')
arg, u, v, label = read_data('../data/{}/{}-{}.npz'.format(type, n, ds_parameter))
nx, ny, dt, T, label_dim, traj_num, step_num, test_index, u, v, label = preprocessing(arg, type, u, v, label, device)
print(u.shape)
print('Checking NS data and models with n = {}, Re = {} ...'.format(nx, Re))


model_ols = UNet([2,4,8,16,32,64,1]).to(device)
model_ed = EDNet().to(device)
model_ed.load_state_dict(torch.load('../models/{}/ED-{}-{}.pth'.format(type, n, Re),
                                    map_location=torch.device('cpu')))
model_ed.eval()
criterion = nn.MSELoss()


#####################################################
# this module aims to see if our flow patterns      #
# change substantially during the simulation        #
#####################################################

'''fig = plt.figure()
ax1 = fig.add_subplot(121)
im1 = ax1.imshow(label[i, 0].reshape(nx, ny))
ax2 = fig.add_subplot(122)
im2 = ax2.imshow(label[i, end].reshape(nx, ny))


print(torch.max(label[i, 0].reshape(nx, ny)))
print(torch.max(label[i, end].reshape(nx, ny)))
#fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=[ax1, ax2])
plt.show()'''


#####################################################
# this module check the accuracy of the learned     # 
# autoencoder, OLS, and TR models                   #
#####################################################
eps = 0.0
batch_size = 1
uv = torch.zeros([batch_size, 2, nx, ny])
#uv[0, 0, :, :] = u[test_index, end].reshape([nx, ny]) + torch.randn(nx, ny) * eps
#uv[0, 1, :, :] = v[test_index, end].reshape([nx, ny]) + torch.randn(nx, ny) * eps
uv[:, 0, :, :] = u[test_index, end:end+batch_size].reshape([batch_size, nx, ny])
uv[:, 1, :, :] = v[test_index, end:end+batch_size].reshape([batch_size, nx, ny])
uv = uv.to(device)
outputs1 = model_ed(uv)
outputs2 = model_ols(uv)
loss_ed = criterion(outputs1, uv)
loss_ols = criterion(outputs2, label[test_index, end:end+batch_size].to(device))


print('Autoencoder Loss: {:4f}'.format(loss_ed.item()))
print('OLS Loss: {:4f}'.format(loss_ols.item()))
print(torch.norm(torch.randn(n, n) * eps, p='fro'))