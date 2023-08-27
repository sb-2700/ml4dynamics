from utils import *
from simulator import *
from models import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch as torch


r.seed(0)
n, beta, Re, type, GPU, ds_parameter = parsing()
device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')

arg, u, v, label = read_data('../data/{}/{}-{}.npz'.format(type, n, ds_parameter))

nx, ny, dt, T, label_dim, traj_num, step_num, test_index, u, v, label = preprocessing(arg, type, u, v, label, device)

print('Training {} model with n = {}, beta = {:.1f}, Re = {} ...'.format(type, n, beta, Re))


# setting training hyperparameters
sample_num = (traj_num-1) * step_num
ed_epochs = 0
ols_epochs = 0
tr_epochs = 100
batch_size = 950
learning_rate = 1e-4
factor = 0.8            # learning rate decay factor
write_interval = 1
period = 2              # related to the scheduler
lambda_ = torch.tensor(1000, requires_grad=False).to(device)


# this part can be simplify to a "load model module"
if type == 'RD':
    model_ols = UNet().to(device)
else:
    model_ols = UNet([2,4,8,16,32,64,1]).to(device)
if os.path.isfile('../models/{}/OLS-{}-{}.pth'.format(type, n, ds_parameter)):
    model_ols.load_state_dict(torch.load('../models/{}/OLS-{}-{}.pth'.format(type, n, ds_parameter), map_location=torch.device('cpu')))
    model_ols.eval()
if type == 'RD':
    model_tr = UNet().to(device)
else:
    model_tr = UNet([2,4,8,16,32,64,1]).to(device)
if os.path.isfile('../models/{}/TR-{}-{}.pth'.format(type, n, ds_parameter)):
    model_tr.load_state_dict(torch.load('../models/{}/TR-{}-{}.pth'.format(type, n, ds_parameter), map_location=torch.device('cpu')))
    model_tr.eval()
# if u switch this EDNet to UNet, the error will become very small
model_ed = EDNet().to(device)
if os.path.isfile('../models/{}/ED-{}-{}.pth'.format(type, n, ds_parameter)):
    model_ed.load_state_dict(torch.load('../models/{}/ED-{}-{}.pth'.format(type, n, ds_parameter), map_location=torch.device('cpu')))
    model_ed.eval()


# Loss and optimizer
criterion = nn.MSELoss()
optimizer_ed = torch.optim.Adam(model_ed.parameters(), lr=learning_rate)
optimizer_ols = torch.optim.Adam(model_ols.parameters(), lr=learning_rate)
optimizer_tr = torch.optim.Adam(model_tr.parameters(), lr=learning_rate)
# maybe try other scheduler
scheduler_ed = ReduceLROnPlateau(optimizer_ed, mode='min', factor=factor, patience=period*sample_num, verbose=False)
scheduler_ols = ReduceLROnPlateau(optimizer_ols, mode='min', factor=factor, patience=period*sample_num, verbose=False)
scheduler_tr = ReduceLROnPlateau(optimizer_tr, mode='min', factor=factor, patience=period*sample_num, verbose=False)


for epoch in range(ed_epochs):
    train_loss = 0
    test_loss = 0
    for j in range(traj_num):
        for i in range(0, step_num-batch_size, batch_size):
            uv = torch.zeros([batch_size, 2, nx, ny])
            uv[:, 0, :, :] = u[j, i:i+batch_size, :].reshape([batch_size, nx, ny])
            uv[:, 1, :, :] = v[j, i:i+batch_size, :].reshape([batch_size, nx, ny])
            uv = uv.to(device)
            outputs = model_ed(uv)
            loss_ed = criterion(outputs, uv)
            if j == test_index:
                # test trajectory is used for validation
                test_loss = test_loss + loss_ed.item()
            else:
                train_loss = train_loss + loss_ed.item()
                optimizer_ed.zero_grad()
                loss_ed.backward()
                optimizer_ed.step()
                scheduler_ed.step(loss_ed)

    if torch.isnan(loss_ed):
        print("Training loss became NaN. Stopping training.")
        break
    if (epoch+1) % write_interval == 0: 
        print ('Autoencoder Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}' 
                .format(epoch+1, ed_epochs, 
                        train_loss*batch_size/step_num/(traj_num-1), 
                        test_loss*batch_size/step_num))

torch.save(model_ed.state_dict(), '../models/{}/ED-{}-{}.pth'.format(type, n, ds_parameter))

# Train the OLS model
for epoch in range(ols_epochs):
    train_loss = 0
    test_loss = 0
    for j in range(traj_num):
        for i in range(0, step_num-batch_size, batch_size):
            uv = torch.zeros([batch_size, 2, nx, ny])
            uv[:, 0, :, :] = u[j, i:i+batch_size, :].reshape([batch_size, nx, ny])
            uv[:, 1, :, :] = v[j, i:i+batch_size, :].reshape([batch_size, nx, ny])
            uv = uv.to(device)
            outputs = model_ols(uv)
            loss_ols = criterion(outputs, label[j, i:i+batch_size, :, :].reshape(batch_size, label_dim, nx, ny))
            if j == test_index:
                # test trajectory is used for validation
                test_loss = test_loss + loss_ols.item()
            else:
                train_loss = train_loss + loss_ols.item()
                optimizer_ols.zero_grad()
                loss_ols.backward()
                optimizer_ols.step()
                scheduler_ols.step(loss_ols)

    if torch.isnan(loss_ols):
        print("Training loss became NaN. Stopping training.")
        break
    if (epoch+1) % write_interval == 0: 
        print ('OLS Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}' 
                .format(epoch+1, ols_epochs, 
                        train_loss*batch_size/step_num/(traj_num-1), 
                        test_loss*batch_size/step_num))

torch.save(model_ols.state_dict(), '../models/{}/OLS-{}-{}.pth'.format(type, n, ds_parameter))

# Train the TR model
for epoch in range(tr_epochs):
    train_loss = 0
    test_loss = 0
    est_loss = 0
    for j in range(traj_num):
        for i in range(0, step_num-batch_size, batch_size):
            uv = torch.zeros([batch_size, 2, nx, ny])
            uv[:, 0, :, :] = u[j, i:i+batch_size, :].reshape([batch_size, nx, ny])
            uv[:, 1, :, :] = v[j, i:i+batch_size, :].reshape([batch_size, nx, ny])
            uv = uv.to(device)
            uv.requires_grad = True
            outputs = model_tr(uv)
            loss_tr = criterion(outputs, label[j, i:i+batch_size, :, :].reshape(batch_size, label_dim, nx, ny))
            est_tr = loss_tr.item()
            z1         = torch.ones_like(uv).to(device)
            de_outputs = model_ed(uv)
            de_loss = criterion(de_outputs, uv)
            grad_de = torch.autograd.grad(de_loss, uv, create_graph = True)
            sum_ = torch.sum(grad_de[0] * outputs/torch.norm(grad_de[0]))
            loss_tr = loss_tr + lambda_*criterion(sum_, torch.tensor(0.0).to(device))
            if j == test_index:
                # test trajectory is used for validation
                test_loss = test_loss + loss_tr.item()
                est_loss = est_loss + est_tr
            else:
                train_loss = train_loss + loss_tr.item()
                optimizer_tr.zero_grad()
                loss_tr.backward()
                optimizer_tr.step()
                scheduler_tr.step(loss_tr)                 

    if torch.isnan(loss_tr):
        print("Training loss became NaN. Stopping training.")
        break    
    if (epoch+1) % write_interval == 0: 
        print ('TR Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}, Test LS Loss: {:4e}, Test Reg Loss: {:4e}' 
                .format(epoch+1, tr_epochs, 
                        train_loss*batch_size/step_num/(traj_num-1), 
                        test_loss*batch_size/step_num,
                        est_loss*batch_size/step_num,
                        (test_loss-est_loss)*batch_size/step_num))
    #if (epoch+1) % 10 == 0:
    #    pdb.set_trace()

torch.save(model_tr.state_dict(), '../models/{}/TR-{}-{}.pth'.format(type, n, ds_parameter))