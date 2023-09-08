import numpy as np
import numpy.random as r
import torch as torch
from utils import *
from models import *
from matplotlib import pyplot as plt


r.seed(0)
n, beta, Re, type, GPU, ds_parameter = parsing()
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

arg, u, v, label = read_data('../data/{}/{}-{}.npz'.format(type, n, ds_parameter))

flag = True
nx, ny, dt, T, label_dim, traj_num, step_num, u, v, label = preprocessing(arg, type, u, v, label, device, flag=flag)

print('Checking {} model with n = {}, beta = {:.1f}, Re = {} ...'.format(type, n, beta, Re))

pdb.set_trace()

test_index = int(np.floor(r.rand() * 10))

model_ed = EDNet().to(device)
model_ed.load_state_dict(torch.load('../models/{}/ED-{}-{}.pth'.format(type, n, ds_parameter),
                                    map_location=torch.device('cpu')))
model_ed.eval()
if type == 'NS':
    model_ols = UNet([2,4,8,16,32,64,1]).to(device)
else:
    model_ols = UNet()
model_ols.load_state_dict(torch.load('../models/{}/OLS-{}-{}.pth'.format(type, n, ds_parameter),
                                    map_location=torch.device('cpu')))
model_ols.eval()
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
end = 20
batch_size = 1
#nx = 128
#ny = nx
uv = torch.zeros([batch_size, 2, nx, ny])
#uv[0, 0, :, :] = u[test_index, end].reshape([nx, ny]) + torch.randn(nx, ny) * eps
#uv[0, 1, :, :] = v[test_index, end].reshape([nx, ny]) + torch.randn(nx, ny) * eps
uv[:, 0, :, :] = u[test_index, end:end+batch_size].reshape([batch_size, nx, ny])
uv[:, 1, :, :] = v[test_index, end:end+batch_size].reshape([batch_size, nx, ny])
uv = uv.to(device)
outputs1 = model_ed(uv)
outputs2 = model_ols(uv)
loss_ed = criterion(outputs1, uv)
#pdb.set_trace()
loss_ols = criterion(outputs2, label[test_index, end:end+batch_size].reshape([label_dim, nx, ny]).to(device))


print('Autoencoder Loss: {:4f}'.format(loss_ed.item()))
print('OLS Loss: {:4f}'.format(loss_ols.item()))
print(torch.norm(torch.randn(n, n) * eps, p='fro'))


batch_size = 10
for j in range(traj_num):
    for i in range(0, step_num-batch_size, batch_size):
        uv = torch.zeros([batch_size, 2, nx, ny])
        uv[:, 0, :, :] = u[j, i:i+batch_size].reshape([batch_size, nx, ny])
        uv[:, 1, :, :] = v[j, i:i+batch_size].reshape([batch_size, nx, ny])
        uv = uv.to(device)
        


batch_size = 10
for j in range(traj_num):
    for i in range(0, step_num-batch_size, batch_size):
        uv = torch.zeros([batch_size, 2, nx, ny])
        uv[:, 0, :, :] = u[j, i:i+batch_size].reshape([batch_size, nx, ny])
        uv[:, 1, :, :] = v[j, i:i+batch_size].reshape([batch_size, nx, ny])
        uv = uv.to(device)
        outputs = model_ols(uv)
        loss_ols = criterion(outputs, label[j, i:i+batch_size, :, :].reshape(batch_size, label_dim, nx, ny))
        outputs = model_ed(uv)
        loss_ed = criterion(outputs, uv)
        print('{}-th AE Loss: {:4e}; traj Loss: {:4e}'.format(j, loss_ed.item(), loss_ols.item()))
#pdb.set_trace()