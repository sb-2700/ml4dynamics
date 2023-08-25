from utils import *
from simulator import *
from models import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch as torch
import numpy.random as r
import argparse


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--Re', type=int, default=400)
parser.add_argument('--n', type=int, default=64)
parser.add_argument('--type', type=str, default='RD')
parser.add_argument('--GPU', type=int, default=0)
args = parser.parse_args()
beta = args.beta
Re = args.Re
n = args.n
type = args.type
GPU = args.GPU
if type == 'RD':
    class_ = int(beta*10)
else:
    class_ = Re
print('Training {} model with n = {}, beta = {:.1f}, Re = {} ...'.format(type, n, beta, Re))


device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
r.seed(0)


arg, u, v, label = read_data('../data/{}/{}-{}.npz'.format(type, n, class_))
label = label.to(device)
nx, ny, dt, T, label_dim = arg
nx = int(nx)
ny = int(ny)
label_dim = int(label_dim)
if type == 'NS':
    u = u[:, :, 1:-1, 1:-1]
    v = v[:, :, 1:-1, :-1]
traj_num = u.shape[0]
step_num = u.shape[1]
test_index = int(np.floor(r.rand() * traj_num))
sample_num = (traj_num-1) * step_num
ed_epochs = 20
ols_epochs = 2
tr_epochs = 20
learning_rate = 1e-3
factor = 0.8            # learning rate decay factor
write_interval = 2
period = 2              # related to the scheduler
lambda_ = torch.tensor(1, requires_grad=False).to(device)


# add condition, checking whether the model already exists
if type == 'RD':
    model_ols = UNet().to(device)
else:
    model_ols = UNet([2,4,8,16,32,64,1]).to(device)
#model_ols.load_state_dict(torch.load('../models/RD/OLS-n{}beta{:.1f}.pth'.format(n, beta)))
#model_ols.eval()
if type == 'RD':
    model_tr = UNet().to(device)
else:
    model_tr = UNet([2,4,8,16,32,64,1]).to(device)
#model_tr.load_state_dict(torch.load('../models/RD/TR-n{}beta{:.1f}.pth'.format(n, beta)))
#model_tr.eval()
# if u switch this EDNet to UNet, the error will become very small
model_ed = EDNet().to(device)
#model_ed.load_state_dict(torch.load('../models/RD/ED-n{}beta{:.1f}.pth'.format(n, beta)))
#model_ed.eval()


# Loss and optimizer
criterion = nn.MSELoss()
optimizer_ed = torch.optim.Adam(model_ed.parameters(), lr=learning_rate)
optimizer_ols = torch.optim.Adam(model_ols.parameters(), lr=learning_rate)
optimizer_tr = torch.optim.Adam(model_tr.parameters(), lr=learning_rate)
# maybe try other scheduler
scheduler_ed = ReduceLROnPlateau(optimizer_ed, mode='min', factor=factor, patience=period*sample_num, verbose=True)
scheduler_ols = ReduceLROnPlateau(optimizer_ols, mode='min', factor=factor, patience=period*sample_num, verbose=True)
scheduler_tr = ReduceLROnPlateau(optimizer_tr, mode='min', factor=factor, patience=period*sample_num, verbose=True)


for epoch in range(ed_epochs):
    total_loss = 0
    for j in range(traj_num):
        for i in range(step_num):
            uv = torch.zeros([1, 2, nx, ny])
            uv[0, 0, :, :] = u[j, i, :].reshape([nx, ny])
            uv[0, 1, :, :] = v[j, i, :].reshape([nx, ny])
            uv = uv.to(device)
            

            outputs = model_ed(uv)
            loss_ed = criterion(outputs, uv)
            optimizer_ed.zero_grad()
            loss_ed.backward()
            optimizer_ed.step()
            scheduler_ed.step(loss_ed)
            
            total_loss = total_loss + loss_ed.item()

    if (epoch+1) % write_interval == 0: 
        print ('Epoch [{}/{}], Autoencoder Loss: {:.7f}' 
                .format(epoch+1, ed_epochs, total_loss))
    

# Train the OLS model
for epoch in range(ols_epochs):
    total_loss_ols = 0
    for j in range(traj_num):
        for i in range(step_num-1):
            uv = torch.zeros([1, 2, nx, ny])
            uv[0, 0, :, :] = u[j, i, :].reshape([nx, ny])
            uv[0, 1, :, :] = v[j, i, :].reshape([nx, ny])
            uv = uv.to(device)
            outputs = model_ols(uv)
            loss_ols = criterion(outputs, label[j, i, :, :].reshape(1, label_dim, nx, ny))
            if j == test_index:
                # test trajectory is used for validation
                total_loss_ols = total_loss_ols + loss_ols.item()
            else:
                optimizer_ols.zero_grad()
                loss_ols.backward()
                optimizer_ols.step()
                scheduler_ols.step(loss_ols)


    print ('Epoch [{}/{}], OLS Loss: {:.7f}' 
            .format(epoch+1, ols_epochs, total_loss_ols))


# Train the TR model
for epoch in range(tr_epochs):
    total_loss_tr = 0
    est_loss_tr = 0
    for j in range(traj_num):
        for i in range(step_num-1):
            uv = torch.zeros([1, 2, nx, ny])
            uv[0, 0, :, :] = u[j, i, :].reshape([nx, ny])
            uv[0, 1, :, :] = v[j, i, :].reshape([nx, ny])
            uv = uv.to(device)
            uv.requires_grad = True
            outputs = model_tr(uv)
            loss_tr = criterion(outputs, label[j, i, :, :].reshape(1, label_dim, nx, ny))
            est_etr = loss_tr.item()
            z1         = torch.ones_like(uv).to(device)
            de_outputs = model_ed(uv)
            de_loss = criterion(de_outputs, uv)
            grad_de = torch.autograd.grad(de_loss, uv, create_graph = True)
            sum_ = torch.sum(grad_de[0] * outputs/torch.norm(grad_de[0]))
            #pdb.set_trace()
            loss_tr = loss_tr + lambda_*criterion(sum_, torch.tensor(0.0).to(device))
            if j == test_index:
                # test trajectory is used for validation
                total_loss_tr = total_loss_tr + loss_tr.item()
                est_loss_tr = est_loss_tr + est_etr
            else:
                optimizer_tr.zero_grad()
                loss_tr.backward()
                optimizer_tr.step()
                scheduler_tr.step(loss_tr)                
                
        
    if (epoch+1) % write_interval == 0: 
        print ('Epoch [{}/{}], TR Total Loss: {:.7f}, Est Loss: {:.7f}' 
                .format(epoch+1, tr_epochs, total_loss_tr, est_loss_tr))
    

torch.save(model_ols.state_dict(), '../models/{}/OLS-{}-{}.pth'.format(type, n, class_))
torch.save(model_tr.state_dict(), '../models/{}/TR-{}-{}.pth'.format(type, n, class_))
torch.save(model_ed.state_dict(), '../models/{}/ED-{}-{}.pth'.format(type, n, class_))