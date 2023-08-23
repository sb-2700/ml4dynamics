from utils import *
from simulator import *
from model import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch as torch
import numpy.random as r
import argparse


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--beta', type=float, default=0.2)
parser.add_argument('--n', type=int, default=64)
args = parser.parse_args()
beta = args.beta
n = args.n
print('Training RD model with n = {} beta = {:.1f} ...'.format(n, beta))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
r.seed(0)


# need to unify the naming for the data & models
u, v, label = read_data('RD/n{}beta{:.1f}.npz'.format(n, beta))
traj_num = u.shape[0]
step_num = u.shape[1]
test_index = int(np.floor(r.rand() * traj_num))
sample_num = traj_num * step_num
ed_epochs = 1000
ols_epochs = 50
tr_epochs = 1000
write_interval = 50
lambda_ = 1


model_ols = UNet().to(device)
#model_ols.load_state_dict(torch.load('../model/RD/OLS-n{}beta{:.1f}.pth'.format(n, beta)))
#model_ols.eval()
model_tr = UNet().to(device)
#model_tr.load_state_dict(torch.load('../model/RD/TR-n{}beta{:.1f}.pth'.format(n, beta)))
#model_tr.eval()
# if u switch this EDNet to UNet, the error will become very small
model_ed = EDNet().to(device)
#model_ed.load_state_dict(torch.load('../model/RD/ED-n{}beta{:.1f}.pth'.format(n, beta)))
#model_ed.eval()


# Loss and optimizer
learning_rate = 1e-3
criterion = nn.MSELoss()
optimizer_ed = torch.optim.Adam(model_ed.parameters(), lr=learning_rate)
optimizer_ols = torch.optim.Adam(model_ols.parameters(), lr=learning_rate)
optimizer_tr = torch.optim.Adam(model_tr.parameters(), lr=learning_rate)
scheduler_ed = ReduceLROnPlateau(optimizer_ed, mode='min', factor=0.5, patience=10*sample_num, verbose=True)
scheduler_ols = ReduceLROnPlateau(optimizer_ols, mode='min', factor=0.5, patience=10*sample_num, verbose=True)
scheduler_tr = ReduceLROnPlateau(optimizer_tr, mode='min', factor=0.5, patience=10*sample_num, verbose=True)


for epoch in range(ed_epochs):
    total_loss = 0
    for j in range(traj_num):
        for i in range(step_num):
            uv = torch.zeros([1, 2, n, n])
            uv[0, 0, :, :] = u[j, i, :].reshape([n, n])
            uv[0, 1, :, :] = v[j, i, :].reshape([n, n])
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
    

# Train the model
for epoch in range(ols_epochs):
    #optimizer = torch.optim.Adam(rd_model.parameters(), lr=learning_rate/(epoch + 1))
    total_loss_ols = 0
    total_loss_tr = 0
    for j in range(traj_num):
        for i in range(step_num-1):
            if j == test_index:
                # test trajectory is used for validation
                total_loss_ols = total_loss_ols + loss_ols.item()
                total_loss_tr = total_loss_tr + loss_tr.item()
            else:
                uv = torch.zeros([1, 2, n, n])
                uv[0, 0, :, :] = u[j, i, :].reshape([n, n])
                uv[0, 1, :, :] = v[j, i, :].reshape([n, n])
                uv = uv.to(device)
                
                if epoch < ols_epochs:
                    # OLS model
                    outputs = model_ols(uv)
                    loss_ols = criterion(outputs, label[j, i, :, :].reshape(1, 2, n, n))
                    optimizer_ols.zero_grad()
                    loss_ols.backward()
                    optimizer_ols.step()
                    scheduler_ols.step(loss_ols)

                
                # TR model
                uv.requires_grad = True
                outputs = model_tr(uv)
                loss_tr = criterion(outputs, label[j, i, :, :].reshape(1, 2, n, n))
                z1         = torch.ones_like(uv)
                de_outputs = model_ed(uv)
                de_loss = criterion(de_outputs, uv)
                grad_de = torch.autograd.grad(de_loss, uv, create_graph = True)
                sum_ = torch.sum(grad_de[0] * outputs/torch.norm(grad_de[0]))
                loss_tr = loss_tr + lambda_*criterion(sum_, torch.Tensor([0]))
                optimizer_tr.zero_grad()
                loss_tr.backward()
                optimizer_tr.step()
                scheduler_tr.step(loss_tr)

        
    if (epoch+1) % write_interval == 0: 
        print ('Epoch [{}/{}], Autoencoder Loss: {:.7f}' 
                .format(epoch+1, ols_epochs, total_loss_ols))
        print ('Epoch [{}/{}], Autoencoder Loss: {:.7f}' 
                .format(epoch+1, ed_epochs, total_loss_tr))
    

torch.save(model_ols.state_dict(), '../model/RD/OLS-n{}beta{:.1f}.pth'.format(n, beta))
torch.save(model_tr.state_dict(), '../model/RD/TR-n{}beta{:.1f}.pth'.format(n, beta))
torch.save(model_ed.state_dict(), '../model/RD/ED-n{}beta{:.1f}.pth'.format(n, beta))