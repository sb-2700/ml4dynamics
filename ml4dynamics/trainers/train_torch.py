import os
import time

import h5py
import ml_collections
import numpy as np
import torch
import torch.nn as nn
import yaml
from box import Box
from ml4dynamics.models.models import Autoencoder, UNet
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

np.set_printoptions(precision=15)
torch.set_default_dtype(torch.float64)

def train(config_dict: ml_collections.ConfigDict):

  config = Box(config_dict)
  pde_type = config.name
  alpha = config.react_diff.alpha
  beta = config.react_diff.beta
  gamma = config.react_diff.gamma
  T = config.react_diff.T
  dt = config.react_diff.dt
  step_num = int(T / dt)
  nx = config.react_diff.nx
  case_num = config.sim.case_num
  batch_size = config.train.batch_size_ae
  # rng = np.random.PRNGKey(config.sim.seed)
  dataset = "alpha{:.2f}_beta{:.2f}_gamma{:.2f}_n{}".format(
    alpha, beta, gamma, case_num
  )
  if pde_type == "react_diff":
    h5_filename = f"data/react_diff/{dataset}.h5"
  GPU = 0
  device = torch.device(
    "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"
  )

  with h5py.File(h5_filename, "r") as h5f:
    inputs = torch.tensor(
      h5f["data"]["inputs"][()], dtype=torch.float64
    ).to(device)
    outputs = torch.tensor(
      h5f["data"]["inputs"][()], dtype=torch.float64
    ).to(device)
  print(
    f"Training {pde_type} model with data: {dataset} ..."
  )
  train_x, test_x, train_y, test_y = train_test_split(
    inputs, outputs, test_size=0.2, random_state=config.sim.seed
  )
  train_dataset = TensorDataset(train_x, train_y)
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_dataset = TensorDataset(test_x, test_y)
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

  # setting training hyperparameters
  sample_num = case_num * step_num
  ae_epochs = config.train.epochs_ae
  ae_epochs = 200
  ols_epochs = 200
  mols_epochs = 200
  aols_epochs = 200
  tr_epochs = 200
  batch_size = step_num
  learning_rate = 1e-4
  factor = 0.8  # learning rate decay factor
  noise_scale = 1e-3  # parameter for adverserial ols
  printInterval = 100
  saveInterval = 100
  period = 2  # related to the scheduler
  lambda_ = torch.tensor(1000, requires_grad=False).to(device)

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
  if os.path.isfile("ckpts/{}/ols-{}.pth".format(pde_type, dataset)):
    model_ols.load_state_dict(
      torch.load(
        "ckpts/{}/ols-{}.pth".format(pde_type, dataset),
        map_location=torch.device("cpu")
      )
    )
    model_ols.eval()
  if os.path.isfile("ckpts/{}/ols-{}.pth".format(pde_type, dataset), ):
    model_mols.load_state_dict(
      torch.load(
        "ckpts/{}/ols-{}.pth".format(pde_type, dataset),
        map_location=torch.device("cpu")
      )
    )
    model_mols.eval()
  if os.path.isfile("ckpts/{}/aols-{}.pth".format(pde_type, dataset)):
    model_aols.load_state_dict(
      torch.load(
        "ckpts/{}/aols-{}.pth".format(pde_type, dataset),
        map_location=torch.device("cpu")
      )
    )
    model_aols.eval()
  if os.path.isfile("ckpts/{}/tr-{}.pth".format(pde_type, dataset)):
    model_tr.load_state_dict(
      torch.load(
        "ckpts/{}/tr-{}.pth".format(pde_type, dataset),
        map_location=torch.device("cpu")
      )
    )
    model_tr.eval()
  # if u switch this Autoencoder to UNet, the error will become very small
  model_ae = Autoencoder(channel_array=[2, 4, 8, 16, 32, 64]).to(device)
  if os.path.isfile("ckpts/{}/ae-{}.pth".format(pde_type, dataset)):
    model_ae.load_state_dict(
      torch.load(
        "ckpts/{}/ae-{}.pth".format(pde_type, dataset),
        map_location=torch.device("cpu")
      )
    )
    model_ae.eval()

  # Loss and optimizer
  criterion = nn.MSELoss()
  optimizer_ae = torch.optim.Adam(model_ae.parameters(), lr=learning_rate)
  optimizer_ols = torch.optim.Adam(model_ols.parameters(), lr=learning_rate)
  optimizer_mols = torch.optim.Adam(
    model_mols.parameters(), lr=learning_rate, weight_decay=0.01
  )
  optimizer_aols = torch.optim.Adam(model_aols.parameters(), lr=learning_rate)
  optimizer_tr = torch.optim.Adam(model_tr.parameters(), lr=learning_rate)
  # maybe try other scheduler
  scheduler_ae = ReduceLROnPlateau(
    optimizer_ae,
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
  iters = tqdm(range(ae_epochs))
  for step in iters:
    for batch_inputs, _ in train_dataloader:
      predict = model_ae(batch_inputs)
      loss_ae = criterion(predict, batch_inputs)
      optimizer_ae.zero_grad()
      loss_ae.backward()
      optimizer_ae.step()
      scheduler_ae.step(loss_ae)
      desc_str = f"{loss_ae.item()=:.4e}"
      iters.set_description_str(desc_str)

    if np.isnan(loss_ae.item()):
      print("Training loss became NaN. Stopping training.")
      break
    # if (epoch + 1) % printInterval == 0:
    #   print(
    #     "Autoencoder Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}".
    #     format(
    #       epoch + 1, ae_epochs,
    #       train_loss * batch_size / step_num / (case_num - 1),
    #       test_loss * batch_size / step_num
    #     )
    #   )
    if (step + 1) % saveInterval == 0:
      torch.save(
        model_ae.state_dict(), "ckpts/{}/ae-{}.pth".format(pde_type, dataset)
      )

  T2 = time.perf_counter()
  print("Training time for Autoencoder model: {:4e}".format(T2 - T1))
  del optimizer_ae, scheduler_ae, loss_ae

  # Train the ols model
  iters = tqdm(range(ols_epochs))
  for _ in iters:
    for batch_inputs, batch_outputs in train_dataloader:
      predict = model_ols(batch_inputs)
      loss_ols = criterion(predict, batch_outputs)
      optimizer_ols.zero_grad()
      loss_ols.backward()
      optimizer_ols.step()
      scheduler_ols.step(loss_ols)
      desc_str = f"{loss_ols.item()=:.4e}"
      iters.set_description_str(desc_str)

    if np.isnan(loss_ols.item()):
      print("Training loss became NaN. Stopping training.")
      break
    # if (epoch + 1) % printInterval == 0:
    #   print(
    #     "ols Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}".format(
    #       epoch + 1, ols_epochs,
    #       train_loss * batch_size / step_num / (case_num - 1),
    #       test_loss * batch_size / step_num
    #     )
    #   )
    if (_ + 1) % saveInterval == 0:
      torch.save(
        model_ols.state_dict(), "ckpts/{}/ols-{}.pth".format(pde_type, dataset)
      )

  T3 = time.perf_counter()
  print("Training time for ols model: {:4e}".format(T3 - T2))
  del model_ols, optimizer_ols, scheduler_ols, loss_ols

  # Train the mols model
  iters = tqdm(range(mols_epochs))
  for _ in iters:
    for batch_inputs, batch_outputs in train_dataloader:
      predict = model_mols(batch_inputs)
      loss_mols = criterion(predict, batch_outputs)
      optimizer_mols.zero_grad()
      loss_mols.backward()
      optimizer_mols.step()
      scheduler_mols.step(loss_mols)
      desc_str = f"{loss_mols.item()=:.4e}"
      iters.set_description_str(desc_str)

    if np.isnan(loss_mols.item()):
      print("Training loss became NaN. Stopping training.")
      break
    # if (epoch + 1) % printInterval == 0:
    #   print(
    #     "mols Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}".format(
    #       epoch + 1, mols_epochs,
    #       train_loss * batch_size / step_num / (case_num - 1),
    #       test_loss * batch_size / step_num
    #     )
    #   )
    if (_ + 1) % saveInterval == 0:
      torch.save(
        model_mols.state_dict(),
        "ckpts/{}/mols-{}.pth".format(pde_type, dataset)
      )

  T4 = time.perf_counter()
  print("Training time for mols model: {:4e}".format(T4 - T3))
  del model_mols, optimizer_mols, scheduler_mols, loss_mols

  # Train the aols model
  iters = tqdm(range(aols_epochs))
  for _ in iters:
    for batch_inputs, batch_outputs in train_dataloader:
      noise = torch.randn(batch_inputs.shape).to(device) * noise_scale
      predict = model_aols(batch_inputs + noise)
      loss_aols = criterion(predict, batch_outputs)
      optimizer_aols.zero_grad()
      loss_aols.backward()
      optimizer_aols.step()
      scheduler_aols.step(loss_aols)
      desc_str = f"{loss_aols.item()=:.4e}"
      iters.set_description_str(desc_str)

    if np.isnan(loss_aols.item()):
      print("Training loss became NaN. Stopping training.")
      break          

    # if (epoch + 1) % printInterval == 0:
    #   print(
    #     "aols Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}".format(
    #       epoch + 1, aols_epochs,
    #       train_loss * batch_size / step_num / (case_num - 1),
    #       test_loss * batch_size / step_num
    #     )
    #   )
    if (_ + 1) % saveInterval == 0:
      torch.save(
        model_aols.state_dict(),
        "ckpts/{}/aols-{}.pth".format(pde_type, dataset)
      )

  T5 = time.perf_counter()
  print("Training time for aols model: {:4e}".format(T5 - T4))
  del model_aols, optimizer_aols, scheduler_aols, loss_aols

  # Train the tr model
  iters = tqdm(range(tr_epochs))
  for _ in iters:
    for batch_inputs, batch_outputs in train_dataloader:
      batch_inputs.requires_grad = True
      predict = model_tr(batch_inputs)
      loss_tr = criterion(predict, batch_outputs)
      de_outputs = model_ae(batch_inputs)
      de_loss = criterion(de_outputs, batch_inputs)
      grad_de = torch.autograd.grad(de_loss, batch_inputs, create_graph=True)
      sum_ = torch.sum(grad_de[0] * predict / torch.norm(grad_de[0]))
      loss_tr = loss_tr + lambda_ * criterion(
        sum_,
        torch.tensor(0.0).to(device)
      )
      optimizer_tr.zero_grad()
      loss_tr.backward()
      optimizer_tr.step()
      scheduler_tr.step(loss_tr)
      desc_str = f"{loss_tr.item()=:.4e}"
      iters.set_description_str(desc_str)

    if np.isnan(loss_tr.item()):
      print("Training loss became NaN. Stopping training.")
      break   
    # if (epoch + 1) % printInterval == 0:
    #   print(
    #     "tr Epoch [{}/{}], Train Loss: {:.4e}, Test Loss: {:4e}, Test LS Loss: {:4e}, Test Reg Loss: {:4e}"
    #     .format(
    #       epoch + 1, tr_epochs,
    #       train_loss * batch_size / step_num / (case_num - 1),
    #       test_loss * batch_size / step_num, est_loss * batch_size / step_num,
    #       (test_loss - est_loss) * batch_size / step_num
    #     )
    #   )
    if (_ + 1) % saveInterval == 0:
      torch.save(
        model_tr.state_dict(), "ckpts/{}/tr-{}.pth".format(pde_type, dataset)
      )

  T6 = time.perf_counter()
  print("Training time for tr model: {:4e}".format(T6 - T5))


if __name__ == "__main__":

  with open("config/simulation.yaml", "r") as file:
    config_dict = yaml.safe_load(file)
  train(config_dict)
