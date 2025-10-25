import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch
# 导入 DataLoader
from torch.utils.data import TensorDataset, DataLoader

# --- 1. 设置设备 (GPU 或 CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. 加载数据 (和以前一样) ---
with open('data/tv_sim_split_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('data/tv_sim_split_valid.pkl', 'rb') as f:
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
    
# --- 3. 关键优化：将数据列表一次性转换为张量 ---
# 这是解决 UserWarning 的核心
print("Converting training list to single tensor...")
# 首先转换为一个大的 numpy 数组，然后转换为 Pytorch 张量，并立刻展平
X_ohe_train_tensor = torch.FloatTensor(np.array(X_ohe_train_list)).view(-1, 1540)

print("Converting validation list to single tensor...")
X_ohe_valid_tensor = torch.FloatTensor(np.array(X_ohe_valid_list)).view(-1, 1540)

print(f"Train tensor shape: {X_ohe_train_tensor.shape}")
print(f"Valid tensor shape: {X_ohe_valid_tensor.shape}")

# --- 4. 定义模型 (和以前一样) ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1540, 512)
        self.fc21 = nn.Linear(512, 32)
        self.fc22 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 512)
        self.fc4 = nn.Linear(512, 1540)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # 移除了 x.view(-1, 1540)，因为传入的 x 已经是正确的形状
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # 移除了 x.view(-1, 1540)，因为 x 已经是正确的形状
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# --- 5. 设置 DataLoader ---
batch_size = 128
epochs = 35

# 使用 TensorDataset 包装张量
train_dataset = TensorDataset(X_ohe_train_tensor)
valid_dataset = TensorDataset(X_ohe_valid_tensor)

# 使用 DataLoader 来自动处理批次、打乱和多进程加载
# num_workers > 0 会启用多进程加载数据，速度更快
# pin_memory=True 在使用 GPU 时能轻微提速
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4, # 尝试设置 4 或 8
    pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

# --- 6. 实例化模型并移动到 GPU ---
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)  

# --- 7. 训练与验证循环 (已优化) ---
f = open("data/loss_w1_optimized.txt", "a")
for epoch in range(epochs):
    # --- 训练 ---
    model.train() # 设置为训练模式
    train_loss = 0
    
    # 使用 DataLoader 迭代，tqdm(train_loader) 会显示进度条
    # (batch_X,) 解包来自 TensorDataset 的元组
    for (batch_X,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        
        # --- 关键：将数据批次移动到 GPU ---
        batch_X = batch_X.to(device)
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch_X) 
        
        # 确保 loss_function 中的 x 也是在 GPU 上的
        loss = loss_function(recon_batch, batch_X, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    f.write(f"Epoch: {epoch}. Train Loss: {avg_train_loss}\n")
    print(f"Epoch: {epoch}. Train Loss: {avg_train_loss}")
    
    # 保存模型
    torch.save(model.state_dict(), f"model/vae_self_tv_sim_split_kl_weight_1_batch_size_{batch_size}_epochs{epoch}.chkpt")

    # --- 验证 (在每个 epoch 训练后立刻进行) ---
    model.eval() # 设置为评估模式
    with torch.no_grad(): # 评估时不需要计算梯度
        valid_loss = 0
        for (batchv_X,) in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
            
            # --- 关键：将数据批次移动到 GPU ---
            batchv_X = batchv_X.to(device)
            
            recon_batch, mu, logvar = model(batchv_X)
            loss = loss_function(recon_batch, batchv_X, mu, logvar)
            valid_loss += loss.item()
    
        avg_valid_loss = valid_loss / len(valid_loader.dataset)
        f.write(f"Epoch: {epoch}. Valid Loss: {avg_valid_loss}\n")
        print(f"Epoch: {epoch}. Valid Loss: {avg_valid_loss}")

f.close()
print("训练和评估结束")

