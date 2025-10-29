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
import torch.optim.lr_scheduler as lr_scheduler # 导入学习率调度器
import logging

import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# 获取当前脚本 (train.py) 的目录
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
# 将项目根目录添加到 Python 搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logging.info(f"已将 {project_root} 添加到 sys.path")


import kaiwu as kw
kw.license.init(user_id="105879747841515522", sdk_code="4vCbDDWqIdUEXDdEHKK0L4MtOOXvMF")

# --- 导入 QVAE 相关的库 ---

from kaiwu_torch_plugin import QVAE, BoltzmannMachine, RestrictedBoltzmannMachine
#  采样器 sampler 在 kaiwu.classical 包中
from kaiwu.classical import SimulatedAnnealingOptimizer

# --- 1. 设置与常量 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {device}")

# --- 关键超参数 ---
INPUT_DIM = 1540  # 70 * 22
LATENT_DIM = 32   # 32 维隐空间
BATCH_SIZE = 2048 # 增加 Batch Size 以利用 5060 Ti 的 16GB 显存
LEARNING_RATE = 2e-4
EPOCHS = 35
NUM_WORKERS = 4   # 增加数据加载器的工作进程

# RBM (先验) 设置
prior_vis = LATENT_DIM // 2
prior_hid = LATENT_DIM - prior_vis

# 确保目录存在
log_save_dir = "data/qvae-v/model"
model_save_dir = "model/qvae-v"
os.makedirs(log_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# 日志文件路径 (附加模式)
log_file_path = os.path.join(log_save_dir, f"loss_qvae_w_optimized_bs{BATCH_SIZE}.txt")

logging.info("按照指示，从 .pkl 文件加载数据...")
with open('data/tv_sim_split_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('data/tv_sim_split_valid.pkl', 'rb') as f:
    X_valid = pickle.load(f)
logging.info("成功加载 .pkl 文件。")

def one_hot_encode(seq):
    mapping = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))     
    seq2 = [mapping[i] for i in seq]
    return np.eye(22)[seq2]

logging.info("One-hot encoding training data...")
X_ohe_train_list = []
for i in tqdm(range(np.shape(X_train)[0])):
    seq = X_train.sequence[i]+'$'
    pad_seq = seq.ljust(70,'0')
    X_ohe_train_list.append(one_hot_encode(pad_seq))

logging.info("One-hot encoding validation data...")
X_ohe_valid_list = []
for i in tqdm(range(np.shape(X_valid)[0])):
    seq = X_valid.sequence[i]+'$'
    pad_seq = seq.ljust(70,'0')
    X_ohe_valid_list.append(one_hot_encode(pad_seq))
    
logging.info("Converting training list to single tensor...")
X_ohe_train_tensor = torch.FloatTensor(np.array(X_ohe_train_list)).view(-1, INPUT_DIM)

mean_x = X_ohe_train_tensor.mean().item()
logging.info(f"Calculated train_bias (mean_x): {mean_x}")

logging.info("Converting validation list to single tensor...")
X_ohe_valid_tensor = torch.FloatTensor(np.array(X_ohe_valid_list)).view(-1, INPUT_DIM)

logging.info(f"Train tensor shape: {X_ohe_train_tensor.shape}")
logging.info(f"Valid tensor shape: {X_ohe_valid_tensor.shape}")

train_dataset = TensorDataset(X_ohe_train_tensor)
valid_dataset = TensorDataset(X_ohe_valid_tensor)

logging.info(f"训练集大小: {len(train_dataset)}")
logging.info(f"验证集大小: {len(valid_dataset)}")

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True # 如果在 GPU 上训练，开启 pin_memory
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# --- 3. 定义 QVAE 模型组件 ---

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

# --- 4. 实例化模型和优化器 ---

# 1. 实例化所有组件
encoder = Encoder(INPUT_DIM, LATENT_DIM).to(device)
decoder = Decoder(LATENT_DIM, INPUT_DIM).to(device)

# 2.  实例化玻尔兹曼机先验 (RBM)
bm_prior = RestrictedBoltzmannMachine(
    num_visible=prior_vis, # 16
    num_hidden=prior_hid   # 16
).to(device)
logging.info(f"已初始化 RestrictedBoltzmannMachine 先验，总共 {bm_prior.num_nodes} 个节点 ({bm_prior.num_visible} 可见 + {bm_prior.num_hidden} 隐藏)。")


logging.info("配置强探索采样器 (SimulatedAnnealingOptimizer)...")
sampler = SimulatedAnnealingOptimizer(
    initial_temperature=500.0,   
    alpha=0.999,                  # 缓慢降温
    cutoff_temperature=0.001,
    iterations_per_t=100,         
    size_limit=100,               
    process_num=-1                
)
# 4. 实例化 QVAE 主模型
model = QVAE(
    encoder=encoder,
    decoder=decoder,
    bm=bm_prior,
    sampler=sampler,
    dist_beta=1.0,           
    mean_x=mean_x, # (修复) 传递 float 类型的 mean_x
    num_vis=bm_prior.num_visible 
).to(device)

# 4.5. 配置优化器和学习率调度器
logging.info(f"配置 Adam 优化器, 学习率: {LEARNING_RATE}")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

logging.info(f"配置 ReduceLROnPlateau 学习率调度器...")
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',         # 监控 'min' (验证损失)
    factor=0.2,         # 学习率衰减为原来的 0.2 倍
    patience=3,         # 3 个 epochs 验证损失没有改善
    min_lr=1e-6         # 最小学习率
)

# --- 5. 训练循环 ---
logging.info(f"--- 开始训练: {EPOCHS} 个 Epochs, Batch Size: {BATCH_SIZE} ---")
with open(log_file_path, 'a') as f: # 'a' = 附加模式
    for epoch in range(EPOCHS):
        
        # --- 训练 ---
        model.train()
        
        # 初始化损失跟踪器
        train_loss_total = 0.0
        train_loss_elbo = 0.0
        train_loss_wd = 0.0

        for (batch_X,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            
            batch_X = batch_X.to(device)
            optimizer.zero_grad()
            
            # 前向传播和损失计算
            (output, recon_x, neg_elbo, wd_loss, 
             total_kl, cost, q, zeta) = model.neg_elbo(batch_X, kl_beta=1.0)
            
            total_loss = neg_elbo + wd_loss
            
            # 反向传播和优化
            total_loss.backward()
            optimizer.step()
            
            # 累加损失
            train_loss_total += total_loss.item()
            train_loss_elbo += neg_elbo.item()
            train_loss_wd += wd_loss.item()
        
        # 计算平均损失
        avg_train_total = train_loss_total / len(train_loader)
        avg_train_elbo = train_loss_elbo / len(train_loader)
        avg_train_wd = train_loss_wd / len(train_loader)
        
        log_msg_train = f"Epoch: {epoch}. Train Loss: {avg_train_total:.4f} (ELBO: {avg_train_elbo:.4f}, WD: {avg_train_wd:.4f})"
        f.write(log_msg_train + "\n")
        logging.info(log_msg_train)
        
        model_save_path = os.path.join(model_save_dir, f"qvae_mts_kl_weight_1_batch_size_{BATCH_SIZE}_epochs{epoch}.chkpt")
        torch.save(model.state_dict(), model_save_path)

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
                
            # 计算平均验证损失
            avg_valid_total = valid_loss_total / len(valid_loader)
            avg_valid_elbo = valid_loss_elbo / len(valid_loader)
            avg_valid_wd = valid_loss_wd / len(valid_loader)
            
            log_msg_valid = f"Epoch: {epoch}. Valid Loss: {avg_valid_total:.4f} (ELBO: {avg_valid_elbo:.4f}, WD: {avg_valid_wd:.4f})"
            f.write(log_msg_valid + "\n")
            logging.info(log_msg_valid)
            
            # --- 更新学习率调度器 ---
            scheduler.step(avg_valid_total)
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f"Epoch: {epoch}. 当前学习率: {current_lr}")
            f.write(f"Epoch: {epoch}. 当前学习率: {current_lr}\n")

logging.info("--- QVAE 训练和评估结束 ---")

