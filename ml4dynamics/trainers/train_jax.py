import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from flax import serialization
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import time

# 创建训练状态
def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((1, 2)))['params']  # 初始化参数
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

# 定义训练步骤
@jax.jit
def train_step(state, batch, model_type="ae"):
    inputs, outputs = batch

    def loss_fn(params):
        pred = state.apply_fn({'params': params}, inputs)
        if model_type == "tr":
            de_outputs = state.apply_fn({'params': params}, inputs)
            grad_de = jax.grad(lambda x: jnp.mean((de_outputs - x) ** 2))(inputs)
            sum_ = jnp.sum(grad_de * pred / jnp.linalg.norm(grad_de))
            reg_loss = jnp.mean(sum_ ** 2)
            return jnp.mean((pred - outputs) ** 2) + 1000 * reg_loss
        else:
            return jnp.mean((pred - outputs) ** 2)

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

# 加载数据
def load_data(file_path):
    with h5py.File(file_path, "r") as h5f:
        inputs = jnp.array(h5f["data"]["inputs"][()])
        outputs = jnp.array(h5f["data"]["outputs"][()])
    return inputs, outputs

# 主训练函数
def train(config):
    # 加载数据
    inputs, outputs = load_data(config["data_path"])
    train_x, test_x, train_y, test_y = train_test_split(
        inputs, outputs, test_size=0.2, random_state=config["seed"]
    )

    # 初始化模型
    rng = jax.random.PRNGKey(config["seed"])
    model_ae = Autoencoder(channels=[2, 4, 8, 16, 32, 64])
    model_ols = UNet(channels=[2, 4, 8, 32, 64, 128, 1])
    model_mols = UNet(channels=[2, 4, 8, 32, 64, 128, 1])
    model_aols = UNet(channels=[2, 4, 8, 32, 64, 128, 1])
    model_tr = UNet(channels=[2, 4, 8, 32, 64, 128, 1])

    # 创建训练状态
    state_ae = create_train_state(rng, model_ae, config["learning_rate"])
    state_ols = create_train_state(rng, model_ols, config["learning_rate"])
    state_mols = create_train_state(rng, model_mols, config["learning_rate"])
    state_aols = create_train_state(rng, model_aols, config["learning_rate"])
    state_tr = create_train_state(rng, model_tr, config["learning_rate"])

    # 训练 Autoencoder
    print("Training Autoencoder...")
    for epoch in tqdm(range(config["ae_epochs"])):
        state_ae = train_step(state_ae, (train_x, train_x), model_type="ae")
        if (epoch + 1) % config["save_interval"] == 0:
            with open(f"ckpts/{config['pde_type']}/ae-{config['dataset']}.pkl", "wb") as f:
                f.write(serialization.to_bytes(state_ae.params))

    # 训练 OLS
    print("Training OLS...")
    for epoch in tqdm(range(config["ols_epochs"])):
        state_ols = train_step(state_ols, (train_x, train_y), model_type="ols")
        if (epoch + 1) % config["save_interval"] == 0:
            with open(f"ckpts/{config['pde_type']}/ols-{config['dataset']}.pkl", "wb") as f:
                f.write(serialization.to_bytes(state_ols.params))

    # 训练 MOLS
    print("Training MOLS...")
    for epoch in tqdm(range(config["mols_epochs"])):
        state_mols = train_step(state_mols, (train_x, train_y), model_type="mols")
        if (epoch + 1) % config["save_interval"] == 0:
            with open(f"ckpts/{config['pde_type']}/mols-{config['dataset']}.pkl", "wb") as f:
                f.write(serialization.to_bytes(state_mols.params))

    # 训练 AOLS
    print("Training AOLS...")
    for epoch in tqdm(range(config["aols_epochs"])):
        state_aols = train_step(state_aols, (train_x, train_y), model_type="aols")
        if (epoch + 1) % config["save_interval"] == 0:
            with open(f"ckpts/{config['pde_type']}/aols-{config['dataset']}.pkl", "wb") as f:
                f.write(serialization.to_bytes(state_aols.params))

    # 训练 TR
    print("Training TR...")
    for epoch in tqdm(range(config["tr_epochs"])):
        state_tr = train_step(state_tr, (train_x, train_y), model_type="tr")
        if (epoch + 1) % config["save_interval"] == 0:
            with open(f"ckpts/{config['pde_type']}/tr-{config['dataset']}.pkl", "wb") as f:
                f.write(serialization.to_bytes(state_tr.params))

# 配置文件
config = {
    "data_path": "data/react_diff/dataset.h5",
    "pde_type": "react_diff",
    "dataset": "alpha1.00_beta1.00_gamma1.00_n1000",
    "seed": 42,
    "learning_rate": 1e-4,
    "ae_epochs": 200,
    "ols_epochs": 200,
    "mols_epochs": 200,
    "aols_epochs": 200,
    "tr_epochs": 200,
    "save_interval": 100,
}

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs(f"ckpts/{config['pde_type']}", exist_ok=True)
    train(config)