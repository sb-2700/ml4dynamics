import os
import time

import ml_collections
import numpy as np
import numpy.random as random
import torch as torch
import yaml
from box import Box
from models import EDNet, UNet
from simulator import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import read_and_preprocess

np.set_printoptions(precision=15)
torch.set_default_dtype(torch.float64)


def train(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  pde_type = config.data.type
  folder = config.data.folder
  dataset = config.data.dataset
  GPU = 0
  device = torch.device(
    "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"
  )

  nx, ny, label_dim, traj_num, step_num, input, output = read_and_preprocess(
    folder + dataset, device
  )
  print(
    "Training {} model with data: {} ...".format(pde_type, config.data.folder)
  )

  # setting training hyperparameters
  sample_num = (traj_num - 1) * step_num
  ed_epochs = 2000
  ols_epochs = 1000
  mols_epochs = 1000
  aols_epochs = 1000
  tr_epochs = 1000
  batch_size = step_num
  learning_rate = 1e-4
  factor = 0.8  # learning rate decay factor
  noise_scale = 1e-3  # paramet.er for adverserial OLS
  printInterval = 100
  saveInterval = 100
  period = 2  # related to the scheduler
  lambda_ = torch.tensor(1000, requires_grad=False).to(device)
  test_index = int(np.floor(r.rand() * 10))

  # this part can be simplify to a "load model module"
  if pde_type == "react_diff":
    model_ols = UNet().to(device)
    model_mols = UNet().to(device)
    model_aols = UNet().to(device)
    model_tr = UNet().to(device)
  else:
    model_ols = UNet([2, 4, 8, 32, 64, 128, 1]).to(device)
    model_mols = UNet([2, 4, 8, 32, 64, 128, 1]).to(device)
    model_aols = UNet([2, 4, 8, 32, 64, 128, 1]).to(device)
    model_tr = UNet([2, 4, 8, 32, 64, 128, 1]).to(device)
  if os.path.isfile("ckpts/{}/OLS-{}.pth".format(pde_type, dataset)):
    model_ols.load_state_dict(
      torch.load(
        "ckpts/{}/OLS-{}.pth".format(pde_type, dataset),
        map_location=torch.device("cpu")
      )
    )
    model_ols.eval()
  if os.path.isfile("ckpts/{}/OLS-{}.pth".format(pde_type, dataset), ):
    model_mols.load_state_dict(
      torch.load(
        "ckpts/{}/OLS-{}.pth".format(pde_type, dataset),
        map_location=torch.device("cpu")
      )
    )
    model_mols.eval()
  if os.path.isfile("ckpts/{}/aOLS-{}.pth".format(pde_type, dataset)):
    model_aols.load_state_dict(
      torch.load(
        "ckpts/{}/aOLS-{}.pth".format(pde_type, dataset),
        map_location=torch.device("cpu")
      )
    )
    model_aols.eval()
  if os.path.isfile("ckpts/{}/TR-{}.pth".format(pde_type, dataset)):
    model_tr.load_state_dict(
      torch.load(
        "ckpts/{}/TR-{}.pth".format(pde_type, dataset),
        map_location=torch.device("cpu")
      )
    )
    model_tr.eval()
  # if u switch this EDNet to UNet, the error will become very small
  model_ed = EDNet(channel_array=[2, 4, 8, 16, 32, 64]).to(device)
  if os.path.isfile("ckpts/{}/ED-{}.pth".format(pde_type, dataset)):
    model_ed.load_state_dict(
      torch.load(
        "ckpts/{}/ED-{}.pth".format(pde_type, dataset),
        map_location=torch.device("cpu")
      )
    )
    model_ed.eval()

  # Loss and optimizer
  criterion = nn.MSELoss()
  optimizer_ed = torch.optim.Adam(model_ed.parameters(), lr=learning_rate)
  optimizer_ols = torch.optim.Adam(model_ols.parameters(), lr=learning_rate)
  optimizer_mols = torch.optim.Adam(
    model_mols.parameters(), lr=learning_rate, weight_decay=0.01
  )
  optimizer_aols = torch.optim.Adam(model_aols.parameters(), lr=learning_rate)
  optimizer_tr = torch.optim.Adam(model_tr.parameters(), lr=learning_rate)
  # maybe try other scheduler
  scheduler_ed = ReduceLROnPlateau(
    optimizer_ed,
    mode="min",
    factor=factor,
    patience=period * sample_num,
    verbose=False
  )
  scheduler_ols = ReduceLROnPlateau(
    optimizer_ols,
    mode="min",
    factor=factor,
    patience=period * sample_num,
    verbose=False
  )
  scheduler_mols = ReduceLROnPlateau(
    optimizer_mols,
    mode="min",
    factor=factor,
    patience=period * sample_num,
    verbose=False
  )
  scheduler_aols = ReduceLROnPlateau(
    optimizer_aols,
    mode="min",
    factor=factor,
    patience=period * sample_num,
    verbose=False
  )
  scheduler_tr = ReduceLROnPlateau(
    optimizer_tr,
    mode="min",
    factor=factor,
    patience=period * sample_num,
    verbose=False
  )

  T1 = time.perf_counter()
  iters = tqdm(range(ed_epochs))
  for epoch in iters:
    train_loss = 0
    test_loss = 0
    for j in range(traj_num):
      for i in range(0, step_num, batch_size):
        uv = input[j, i:i + batch_size].reshape([batch_size, 2, nx,
                                                 ny]).to(device)
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

    if np.isnan(train_loss).item():
      print("Training loss became NaN. Stopping training.")
      break
    if (epoch + 1) % printInterval == 0:
      print(
        "Autoencoder Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}".
        format(
          epoch + 1, ed_epochs,
          train_loss * batch_size / step_num / (traj_num - 1),
          test_loss * batch_size / step_num
        )
      )
    if (epoch + 1) % saveInterval == 0:
      torch.save(
        model_ed.state_dict(), "ckpts/{}/ED-{}.pth".format(pde_type, dataset)
      )

  T2 = time.perf_counter()
  print("Training time for ED model: {:4e}".format(T2 - T1))
  del optimizer_ed, scheduler_ed, loss_ed

  # Train the OLS model
  iters = tqdm(range(ols_epochs))
  for epoch in iters:
    train_loss = 0
    test_loss = 0
    for j in range(traj_num):
      for i in range(0, step_num, batch_size):
        uv = U[j, i:i + batch_size].reshape([batch_size, 2, nx, ny]).to(device)
        outputs = model_ols(uv)
        loss_ols = criterion(
          outputs,
          output[j,
                 i:i + batch_size, :, :].reshape(batch_size, label_dim, nx, ny)
        )
        if j == test_index:
          # test trajectory is used for validation
          test_loss = test_loss + loss_ols.item()
        else:
          train_loss = train_loss + loss_ols.item()
          optimizer_ols.zero_grad()
          loss_ols.backward()
          optimizer_ols.step()
          scheduler_ols.step(loss_ols)

    if np.isnan(train_loss).item():
      print("Training loss became NaN. Stopping training.")
      break
    if (epoch + 1) % printInterval == 0:
      print(
        "OLS Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}".format(
          epoch + 1, ols_epochs,
          train_loss * batch_size / step_num / (traj_num - 1),
          test_loss * batch_size / step_num
        )
      )
    if (epoch + 1) % saveInterval == 0:
      torch.save(
        model_ols.state_dict(), "ckpts/{}/OLS-{}.pth".format(pde_type, dataset)
      )

  T3 = time.perf_counter()
  print("Training time for OLS model: {:4e}".format(T3 - T2))
  del model_ols, optimizer_ols, scheduler_ols, loss_ols

  # Train the mOLS model
  iters = tqdm(range(mols_epochs))
  for epoch in iters:
    train_loss = 0
    test_loss = 0
    for j in range(traj_num):
      for i in range(0, step_num, batch_size):
        uv = U[j, i:i + batch_size].reshape([batch_size, 2, nx, ny]).to(device)
        outputs = model_mols(uv)
        loss_mols = criterion(
          outputs,
          output[j,
                 i:i + batch_size, :, :].reshape(batch_size, label_dim, nx, ny)
        )
        if j == test_index:
          # test trajectory is used for validation
          test_loss = test_loss + loss_mols.item()
        else:
          train_loss = train_loss + loss_mols.item()
          optimizer_mols.zero_grad()
          loss_mols.backward()
          optimizer_mols.step()
          scheduler_mols.step(loss_mols)

    if np.isnan(train_loss).item():
      print("Training loss became NaN. Stopping training.")
      break
    if (epoch + 1) % printInterval == 0:
      print(
        "mOLS Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}".format(
          epoch + 1, mols_epochs,
          train_loss * batch_size / step_num / (traj_num - 1),
          test_loss * batch_size / step_num
        )
      )
    if (epoch + 1) % saveInterval == 0:
      torch.save(
        model_mols.state_dict(),
        "ckpts/{}/mOLS-{}.pth".format(pde_type, dataset)
      )

  T4 = time.perf_counter()
  print("Training time for mOLS model: {:4e}".format(T4 - T3))
  del model_mols, optimizer_mols, scheduler_mols, loss_mols

  # Train the aOLS model
  iters = tqdm(range(aols_epochs))
  for epoch in iters:
    train_loss = 0
    test_loss = 0
    for j in range(traj_num):
      for i in range(0, step_num, batch_size):
        uv = (
          U[j, i:i + batch_size].reshape([batch_size, 2, nx, ny]) +
          noise_scale * torch.randn(batch_size, 2, nx, ny)
        ).to(device)
        outputs = model_aols(uv)
        loss_aols = criterion(
          outputs,
          output[j,
                 i:i + batch_size, :, :].reshape(batch_size, label_dim, nx, ny)
        )
        if j == test_index:
          # test trajectory is used for validation
          test_loss = test_loss + loss_aols.item()
        else:
          train_loss = train_loss + loss_aols.item()
          optimizer_aols.zero_grad()
          loss_aols.backward()
          optimizer_aols.step()
          scheduler_aols.step(loss_aols)

    if np.isnan(train_loss).item():
      print("Training loss became NaN. Stopping training.")
      break
    if (epoch + 1) % printInterval == 0:
      print(
        "aOLS Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}".format(
          epoch + 1, aols_epochs,
          train_loss * batch_size / step_num / (traj_num - 1),
          test_loss * batch_size / step_num
        )
      )
    if (epoch + 1) % saveInterval == 0:
      torch.save(
        model_aols.state_dict(),
        "ckpts/{}/aOLS-{}.pth".format(pde_type, dataset)
      )

  T5 = time.perf_counter()
  print("Training time for aOLS model: {:4e}".format(T5 - T4))
  del model_aols, optimizer_aols, scheduler_aols, loss_aols

  # Train the TR model
  iters = tqdm(range(tr_epochs))
  for epoch in iters:
    train_loss = 0
    test_loss = 0
    est_loss = 0
    for j in range(traj_num):
      for i in range(0, step_num, batch_size):
        uv = U[j, i:i + batch_size].reshape([batch_size, 2, nx, ny]).to(device)
        uv.requires_grad = True
        outputs = model_tr(uv)
        loss_tr = criterion(
          outputs,
          output[j,
                 i:i + batch_size, :, :].reshape(batch_size, label_dim, nx, ny)
        )
        est_tr = loss_tr.item()
        z1 = torch.ones_like(uv).to(device)
        de_outputs = model_ed(uv)
        de_loss = criterion(de_outputs, uv)
        grad_de = torch.autograd.grad(de_loss, uv, create_graph=True)
        sum_ = torch.sum(grad_de[0] * outputs / torch.norm(grad_de[0]))
        loss_tr = loss_tr + lambda_ * criterion(
          sum_,
          torch.tensor(0.0).to(device)
        )
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

    if np.isnan(train_loss).item():
      print("Training loss became NaN. Stopping training.")
      break
    if (epoch + 1) % printInterval == 0:
      print(
        "TR Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}, Test LS Loss: {:4e}, Test Reg Loss: {:4e}"
        .format(
          epoch + 1, tr_epochs,
          train_loss * batch_size / step_num / (traj_num - 1),
          test_loss * batch_size / step_num, est_loss * batch_size / step_num,
          (test_loss - est_loss) * batch_size / step_num
        )
      )
    if (epoch + 1) % saveInterval == 0:
      torch.save(
        model_tr.state_dict(), "ckpts/{}/TR-{}.pth".format(pde_type, dataset)
      )

  T6 = time.perf_counter()
  print("Training time for TR model: {:4e}".format(T6 - T5))


if __name__ == "__main__":

  with open("config/train.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  train(config_dict)
