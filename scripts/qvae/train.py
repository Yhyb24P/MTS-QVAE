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

# --- TODO: 导入 QVAE 相关的库 ---
# 请根据您的库名称修改 "kaiwu_torch_plugin"
# 这些导入是基于 "MTS-QBM-VAE工作流.md" 的分析
try:
    from kaiwu_torch_plugin import QVAE, RestrictedBoltzmannMachine
    from kaiwu_torch_plugin.qvae_dist_util import FactorialBernoulliUtil
    # QVAE 训练需要一个采样器来估计 log Z
    # 您必须提供一个采样器实现，这里以 SimulatedAnnealingSampler 为例
    from kaiwu_torch_plugin.samplers import SimulatedAnnealingSampler 
except ImportError:
    print("="*50)
    print("错误：无法导入 QVAE 库 (kaiwu_torch_plugin)。")
    print("请确保已安装该库，或已将 'kaiwu_torch_plugin' 替换为正确的包名。")
    print("您还需要一个采样器实现 (例如 SimulatedAnnealingSampler)。")
    print("="*50)
    # 抛出异常以停止执行，因为后续代码无法运行
    raise

# --- 1. 设置与常量 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 统一定义维度
INPUT_DIM = 1540  # 70 * 22
LATENT_DIM = 32   # 保持与您原 VAE 一致的 32 维隐空间
BATCH_SIZE = 128
EPOCHS = 35

# --- 2. 加载数据 (与原脚本相同) ---
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

print("Converting validation list to single tensor...")
X_ohe_valid_tensor = torch.FloatTensor(np.array(X_ohe_valid_list)).view(-1, INPUT_DIM)

print(f"Train tensor shape: {X_ohe_train_tensor.shape}")
print(f"Valid tensor shape: {X_ohe_valid_tensor.shape}")

# --- 3. 定义 QVAE 的组件 (编码器和解码器) ---
# 根据 "MTS-QBM-VAE工作流.md"，我们将原 VAE 拆分为 Encoder 和 Decoder

class Encoder(nn.Module):
    """编码器： p(x) -> q_logits (用于 q(z|x))"""
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        # 不再需要 fc21(mu) 和 fc22(logvar)
        # 而是输出离散后验的 logits
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
        # QVAE 的 z 是二元 (0/1) 的，需要转为 float
        h3 = self.relu(self.fc3(z.float()))
        # 输出 logits，损失函数将处理 sigmoid
        return self.fc4(h3)

# --- 4. 设置 DataLoader (与原脚本相同) ---
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

# 1. 实例化编码器和解码器
encoder = Encoder(INPUT_DIM, LATENT_DIM).to(device)
decoder = Decoder(LATENT_DIM, INPUT_DIM).to(device)

# 2. 实例化玻尔兹曼机先验 (RBM)
# RBM 的可见层 (n_visible) 必须等于您的隐空间维度 (LATENT_DIM)
rbm_prior = RestrictedBoltzmannMachine(
    n_visible=LATENT_DIM, 
    n_hidden=64  # 这是一个超参数，您可以调整
).to(device)

# 3. 实例化采样器 (TODO: 必须提供)
# "工作流" 指出需要采样器来估计 log Z
# 您需要配置您的采样器，例如：
sampler = SimulatedAnnealingSampler(n_steps=100) 

# 4. 实例化 QVAE 主模型
model = QVAE(
    encoder=encoder,
    decoder=decoder,
    bm=rbm_prior,
    recon_dist_util=FactorialBernoulliUtil, # 使用文档中提到的工具
    sampler=sampler
).to(device)

# 5. 实例化优化器
# 优化器将自动管理 QVAE 内部所有参数 (encoder, decoder, bm)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- 6. 训练与验证循环 (已修改为 QVAE) ---
f = open("qdata/loss_qvae_w1.txt", "a")

# 删除旧的 loss_function
# def loss_function(recon_x, x, mu, logvar): ... (已删除)

for epoch in range(EPOCHS):
    # --- 训练 ---
    model.train() # 设置为训练模式
    train_loss = 0
    
    for (batch_X,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        
        batch_X = batch_X.to(device)
        optimizer.zero_grad()
        
        # --- 核心改动：使用 QVAE 的 neg_elbo 作为损失 ---
        # "工作流" 指出 QVAE 的训练是端到端的，通过 neg_elbo 计算
        # kl_beta 是 KL 散度的权重，可以用于退火 (从 0 缓慢增加到 1)
        # 这里为简单起见，我们直接设为 1.0
        kl_beta = 1.0 
        loss = model.neg_elbo(batch_X, kl_beta=kl_beta)
        # --- 结束改动 ---
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    f.write(f"Epoch: {epoch}. Train Loss: {avg_train_loss}\n")
    print(f"Epoch: {epoch}. Train Loss: {avg_train_loss}")
    
    # 保存模型 (使用新名称)
    torch.save(model.state_dict(), f"model/qvae_mts_kl_weight_1_batch_size_{BATCH_SIZE}_epochs{epoch}.chkpt")

    # --- 验证 ---
    model.eval() # 设置为评估模式
    with torch.no_grad():
        valid_loss = 0
        for (batchv_X,) in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]"):
            
            batchv_X = batchv_X.to(device)
            
            # 验证时也使用 neg_elbo
            loss = model.neg_elbo(batchv_X, kl_beta=1.0)
            valid_loss += loss.item()
    
        avg_valid_loss = valid_loss / len(valid_loader.dataset)
        f.write(f"Epoch: {epoch}. Valid Loss: {avg_valid_loss}\n")
        print(f"Epoch: {epoch}. Valid Loss: {avg_valid_loss}")

f.close()
print("QVAE 训练和评估结束")