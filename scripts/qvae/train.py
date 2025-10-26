import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader

import sys
import os
# 获取当前脚本 (train.py) 的目录
# e.g., /home/yhshy/git_files/MTS-QBM-VAE/scripts/qvae
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目的根目录 (向上两级)
# e.g., /home/yhshy/git_files/MTS-QBM-VAE
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
# 将项目根目录添加到 Python 搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"已将 {project_root} 添加到 sys.path")


import kaiwu as kw
kw.license.init(user_id="105879747841515522", sdk_code="4vCbDDWqIdUEXDdEHKK0L4MtOOXvMF")

# --- TODO: 导入 QVAE 相关的库 ---
try:
    #  我们需要 RBM (作为先验) 和 QVAE (主模型)
    from kaiwu_torch_plugin import QVAE, BoltzmannMachine, RestrictedBoltzmannMachine
    #  采样器 sampler 在 kaiwu.classical 包中
    from kaiwu.classical import SimulatedAnnealingOptimizer
except ImportError:
    print("="*50)
    print(f"错误：无法导入 QVAE 库 (kaiwu_torch_plugin)。")
    print(f"已将 {project_root} 添加到 sys.path，请检查该目录下是否存在 'kaiwu_torch_plugin' 文件夹。")
    print("="*50)
    raise

# --- 1. 设置与常量 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


INPUT_DIM = 1540  # 70 * 22
LATENT_DIM = 32   # 保持与原 VAE 一致的 32 维隐空间
BATCH_SIZE = 128
EPOCHS = 35

# --- 2. 加载数据  ---
with open('qdata/tv_sim_split_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('qdata/tv_sim_split_valid.pkl', 'rb') as f:
    X_valid = pickle.load(f)

def one_hot_encode(seq):
    mapping = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(22)[seq2]

print("One-hot encoding training data...")
X_ohe_train_list = []
for i in tqdm(range(np.shape(X_train)[0])):
    seq = X_train.sequence[i]+'$'
    pad_seq = seq.ljust(70,'0')
    X_ohe_train_list.append(one_hot_encode(pad_seq))

print("One-hot encoding validation data...")
X_ohe_valid_list = []
for i in tqdm(range(np.shape(X_valid)[0])):
    seq = X_valid.sequence[i]+'$'
    pad_seq = seq.ljust(70,'0')
    X_ohe_valid_list.append(one_hot_encode(pad_seq))
    
print("Converting training list to single tensor...")
X_ohe_train_tensor = torch.FloatTensor(np.array(X_ohe_train_list)).view(-1, INPUT_DIM)
mean_x = X_ohe_train_tensor.mean().item()
print(f"Calculated train_bias (mean_x): {mean_x}")

print("Converting validation list to single tensor...")
X_ohe_valid_tensor = torch.FloatTensor(np.array(X_ohe_valid_list)).view(-1, INPUT_DIM)

print(f"Train tensor shape: {X_ohe_train_tensor.shape}")
print(f"Valid tensor shape: {X_ohe_valid_tensor.shape}")

# --- 3. 定义 QVAE 的组件 (编码器和解码器) ---
class Encoder(nn.Module):
    """编码器： p(x) -> q_logits (用于 q(z|x))"""
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc_logits = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc_logits(h1)

class Decoder(nn.Module):
    """解码器： z -> recon_logits (用于 p(x|z))"""
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        h3 = self.relu(self.fc3(z.float()))
        return self.fc4(h3)

# --- 4. 设置 DataLoader  ---
train_dataset = TensorDataset(X_ohe_train_tensor)
valid_dataset = TensorDataset(X_ohe_valid_tensor)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

# --- 5. 实例化 QVAE 模型、采样器和优化器 ---

encoder = Encoder(INPUT_DIM, LATENT_DIM).to(device)
decoder = Decoder(LATENT_DIM, INPUT_DIM).to(device)

prior_vis = LATENT_DIM // 2
prior_hid = LATENT_DIM - prior_vis
bm_prior = RestrictedBoltzmannMachine(
    num_visible=prior_vis, # 16
    num_hidden=prior_hid   # 16
).to(device)
print(f"已初始化 RestrictedBoltzmannMachine 先验，总共 {bm_prior.num_nodes} 个节点 ({bm_prior.num_visible} 可见 + {bm_prior.num_hidden} 隐藏)。")

sampler = SimulatedAnnealingOptimizer(alpha=0.999, size_limit=100) 

model = QVAE(
    encoder=encoder,
    decoder=decoder,
    bm=bm_prior,
    sampler=sampler,
    dist_beta=1.0,
    mean_x=mean_x,
    num_vis=bm_prior.num_visible
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- 6. 训练与验证循环  ---
f = open("qdata/loss_qvae_w1.txt", "a")

for epoch in range(EPOCHS):
    # --- 训练 ---
    model.train()
    #  初始化所有损失跟踪器
    train_loss_total = 0.0 # 总损失 (ELBO + WD)
    train_loss_elbo = 0.0  # 仅 ELBO (KL + 重建)
    train_loss_wd = 0.0    # 仅 权重衰减
    
    for (batch_X,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        
        batch_X = batch_X.to(device)
        optimizer.zero_grad()
        
        (output, recon_x, neg_elbo, wd_loss, 
         total_kl, cost, q, zeta) = model.neg_elbo(batch_X, kl_beta=1.0)
        
        #  loss 是 ELBO 和 权重衰减 的总和
        loss = neg_elbo + wd_loss 
        
        loss.backward()

        #  累加所有损失
        train_loss_total += loss.item()
        train_loss_elbo += neg_elbo.item()
        train_loss_wd += wd_loss.item()

        optimizer.step()
    
    #  计算并打印所有平均损失
    avg_train_total = train_loss_total / len(train_loader)
    avg_train_elbo = train_loss_elbo / len(train_loader)
    avg_train_wd = train_loss_wd / len(train_loader)
    
    log_msg_train = f"Epoch: {epoch}. Train Loss: {avg_train_total:.4f} (ELBO: {avg_train_elbo:.4f}, WD: {avg_train_wd:.4f})"
    f.write(log_msg_train + "\n")
    print(log_msg_train)
    
    torch.save(model.state_dict(), f"model/qvae_mts_kl_weight_1_batch_size_{BATCH_SIZE}_epochs{epoch}.chkpt")

    # --- 验证 ---
    model.eval()
    with torch.no_grad():
        #  初始化所有损失跟踪器
        valid_loss_total = 0.0
        valid_loss_elbo = 0.0
        valid_loss_wd = 0.0

        for (batchv_X,) in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]"):
            
            batchv_X = batchv_X.to(device)
            
            (v_output, v_recon_x, v_neg_elbo, v_wd_loss, 
             v_total_kl, v_cost, v_q, v_zeta) = model.neg_elbo(batchv_X, kl_beta=1.0)
            
            #  累加所有损失
            valid_loss_total += (v_neg_elbo + v_wd_loss).item()
            valid_loss_elbo += v_neg_elbo.item()
            valid_loss_wd += v_wd_loss.item()
    
        #  计算并打印所有平均损失
        avg_valid_total = valid_loss_total / len(valid_loader)
        avg_valid_elbo = valid_loss_elbo / len(valid_loader)
        avg_valid_wd = valid_loss_wd / len(valid_loader)

        log_msg_valid = f"Epoch: {epoch}. Valid Loss: {avg_valid_total:.4f} (ELBO: {avg_valid_elbo:.4f}, WD: {avg_valid_wd:.4f})"
        f.write(log_msg_valid + "\n")
        print(log_msg_valid)

f.close()
print("QVAE 训练和评估结束")

