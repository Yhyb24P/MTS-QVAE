import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt #  导入 matplotlib
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch
from torch.utils.data import Dataset, DataLoader
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
INPUT_DIM = 1540  # 70 * 22 (保持不变，因为 CNN 在模块内部处理)
LATENT_DIM = 32   # 32 维隐空间
BATCH_SIZE = 2048 # 增加 Batch Size
LEARNING_RATE = 1e-4 # (保留您最新的学习率)
EPOCHS = 35
NUM_WORKERS = 4   # 数据加载器的工作进程
MAX_LEN = 70      # 序列最大长度
CHANNELS = 22     # 20 个氨基酸 + '$' + '0'

# RBM (先验) 设置
prior_vis = LATENT_DIM // 2
prior_hid = LATENT_DIM - prior_vis

# 确保目录存在
log_save_dir = "data/qvae-v/model"
model_save_dir = "model/qvae-v"
os.makedirs(log_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# 日志文件路径 (附加模式)
log_file_path = os.path.join(log_save_dir, f"loss_qvae_cnn_fixed_bs{BATCH_SIZE}.txt")

# --- 2. 数据加载与预处理 ---

MAPPING = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(CHANNELS)))
def one_hot_encode(seq):
    """将填充/截断后的序列 OHE"""
    seq2 = [MAPPING[i] for i in seq]
    return np.eye(CHANNELS)[seq2]

class SequenceDataset(Dataset):
    """
      自定义数据集，用于即时（on-the-fly）加载和处理数据。
    """
    def __init__(self, pkl_file_path, max_len):
        logging.info(f"Loading data from {pkl_file_path}...")
        with open(pkl_file_path, 'rb') as f:
            self.data = pickle.load(f)
        self.max_len = max_len
        self.pad_char = '0'
        self.end_char = '$'
        logging.info(f"Loaded {len(self.data)} sequences.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 1. 获取原始序列
        seq_raw = self.data.sequence[idx]
        
        # 2.   健壮的截断和填充
        #    先截断，确保最大长度
        seq_terminated = (seq_raw + self.end_char)[:self.max_len]
        #    后填充，确保最小长度
        seq_padded = seq_terminated.ljust(self.max_len, self.pad_char)
        
        # 3.  必须在 CPU (Dataset) 上执行 OHE
        ohe_seq = one_hot_encode(seq_padded) # Shape: (70, 22)
        
        # 4.  必须返回扁平化的 OHE 张量
        ohe_tensor_flat = torch.FloatTensor(ohe_seq).view(-1) # Shape: (1540)
        
        return ohe_tensor_flat # 返回 (1540) 形状的 FloatTensor

def get_mean_x(dataset, batch_size, num_workers):
    """  辅助函数，从即时数据集中计算 mean_x"""
    logging.info("Calculating mean_x from training data...")
    temp_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    total_sum = 0.0
    total_count = 0
    
    for batch in tqdm(temp_loader, desc="Calculating mean_x"):
        total_sum += batch.sum().item()
        total_count += batch.numel()
        
    mean_x = total_sum / total_count
    logging.info(f"Calculated train_bias (mean_x): {mean_x}")
    return mean_x

# --- 3. 实例化数据集和加载器 ---
train_dataset = SequenceDataset(pkl_file_path='data/tv_sim_split_train.pkl', max_len=MAX_LEN)
valid_dataset = SequenceDataset(pkl_file_path='data/tv_sim_split_valid.pkl', max_len=MAX_LEN)

# 计算 mean_x
mean_x = get_mean_x(train_dataset, BATCH_SIZE, NUM_WORKERS)

# --- 将 mean_x 保存到文件 ---
# generate.py 脚本在初始化 QVAE 时需要这个值
mean_x_save_path = os.path.join(model_save_dir, f"mean_x_bs{BATCH_SIZE}.pkl")
try:
    with open(mean_x_save_path, 'wb') as f:
        pickle.dump(mean_x, f)
    logging.info(f"已将 mean_x 保存到: {mean_x_save_path}")
except Exception as e:
    logging.error(f"保存 mean_x 失败: {e}")
# --- 结束修正 ---

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


# --- 4. 定义 QVAE 模型组件   ---

class Encoder(nn.Module):
    """
      编码器：使用 1D-CNN 处理序列结构
    输入: (B, 1540) -> 内部 reshape -> (B, 22, 70)
    输出: (B, latent_dim)
    """
    def __init__(self, input_dim, latent_dim, channels=CHANNELS, seq_len=MAX_LEN):
        super(Encoder, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) # 70 -> 35
        
        # 128 channels * 35 length
        self.flat_dim = 128 * (seq_len // 2) 
        
        self.fc1 = nn.Linear(self.flat_dim, 512)
        self.fc_logits = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x): # x shape: (B, 1540)
        # 1.  Reshape 扁平化的 OHE 张量
        h = x.view(-1, self.channels, self.seq_len) # (B, 22, 70)
        
        # 2. CNN layers
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        h = self.pool(h) # (B, 128, 35)
        
        # 3. Flatten
        h = h.view(-1, self.flat_dim) # (B, 128 * 35)
        
        # 4. FC layers
        h1 = self.relu(self.fc1(h))
        return self.fc_logits(h1)

class Decoder(nn.Module):
    """
      解码器：使用 1D-CNN (ConvTranspose) 重建序列
    输入: (B, latent_dim)
    输出: (B, 1540)
    """
    def __init__(self, latent_dim, output_dim, channels=CHANNELS, seq_len=MAX_LEN):
        super(Decoder, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.start_len = seq_len // 2 # 35
        self.start_flat_dim = 128 * self.start_len # 128 * 35
        
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc_upscale = nn.Linear(512, self.start_flat_dim)
        
        self.unpool = nn.Upsample(scale_factor=2) # 35 -> 70
        self.tconv1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.tconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=channels, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU()

    def forward(self, z): # z shape: (B, latent_dim)
        # 1. FC layers
        h = self.relu(self.fc3(z.float()))
        h = self.relu(self.fc_upscale(h)) # (B, 128 * 35)
        
        # 2. Reshape for CNN
        h = h.view(-1, 128, self.start_len) # (B, 128, 35)
        
        # 3. Transpose CNN
        h = self.unpool(h) # (B, 128, 70)
        h = self.relu(self.tconv1(h)) # (B, 64, 70)
        h = self.tconv2(h) # (B, 22, 70) -> Logits, no activation
        
        # 4. Flatten to match output_dim
        return h.view(-1, self.channels * self.seq_len) # (B, 1540)

# --- 5. 实例化模型和优化器 ---

# 1. 实例化所有组件 (使用新的 CNN Encoder/Decoder)
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
    mean_x=mean_x, #   传递计算好的 mean_x
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
    min_lr=1e-6         # (修正 2): 降低 min_lr 以允许多次衰减
)

# --- 6. 训练循环  ---
logging.info(f"--- 开始训练: {EPOCHS} 个 Epochs, Batch Size: {BATCH_SIZE} ---")
best_valid_loss = float('inf') # (Fix 4b) 跟踪最佳验证损失

#  初始化用于绘图的历史列表
train_history_elbo = []
valid_history_elbo = []
train_history_cost = []
valid_history_cost = []

# : 移除自定义权重设置

with open(log_file_path, 'a') as f: # 'a' = 附加模式
    # : 更新日志
    f.write(f"使用标准 ELBO 损失 (Cost + KL)\n")
    
    for epoch in range(EPOCHS):
        

        # --- 训练 ---
        model.train()
        
        # 初始化损失跟踪器
        train_loss_total = 0.0
        train_loss_elbo = 0.0
        train_loss_wd = 0.0
        train_loss_cost = 0.0 #  跟踪原始重构损失

        # : 移除 (batch_X,) 的元组解包
        for batch_X in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            
            batch_X = batch_X.to(device)
            (output, recon_x, neg_elbo, wd_loss, 
             total_kl, cost, q, zeta) = model.neg_elbo(batch_X, kl_beta=1.0)
            
            # : 移除手动损失计算
            # neg_elbo = (cost_weight * cost) + (kl_weight * total_kl)
            
            # : total_loss 现在直接使用库返回的 neg_elbo
            total_loss = neg_elbo + wd_loss
            
            # 反向传播和优化
            total_loss.backward()
            optimizer.step()
            
            # 累加损失
            train_loss_total += total_loss.item()
            train_loss_elbo += neg_elbo.item()
            train_loss_wd += wd_loss.item()
            train_loss_cost += cost.item() #  累加原始重构损失
        
        # 计算平均损失
        avg_train_total = train_loss_total / len(train_loader)
        avg_train_elbo = train_loss_elbo / len(train_loader)
        avg_train_wd = train_loss_wd / len(train_loader)
        avg_train_cost = train_loss_cost / len(train_loader) #  计算平均重构损失
        
        #  存储历史
        train_history_elbo.append(avg_train_elbo)
        train_history_cost.append(avg_train_cost)
        
        log_msg_train = f"Epoch: {epoch}. Train Loss: {avg_train_total:.4f} (ELBO: {avg_train_elbo:.4f}, Cost: {avg_train_cost:.4f}, WD: {avg_train_wd:.4f})" # (日志)
        f.write(log_msg_train + "\n")
        logging.info(log_msg_train)
        
        # (Fix 4b) 移除在每个 epoch 都保存模型的代码

        # --- 验证 ---
        model.eval()
        with torch.no_grad():
            #  初始化所有损失跟踪器
            valid_loss_total = 0.0
            valid_loss_elbo = 0.0
            valid_loss_wd = 0.0
            valid_loss_cost = 0.0 #  跟踪原始重构损失

            # : 移除 (batchv_X,) 的元组解包
            for batchv_X in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]"):
                
                batchv_X = batchv_X.to(device)
                
                # : 直接使用返回的 v_neg_elbo
                (v_output, v_recon_x, v_neg_elbo, v_wd_loss, 
                 v_total_kl, v_cost, v_q, v_zeta) = model.neg_elbo(batchv_X, kl_beta=1.0)
                
                
                #  累加所有损失
                valid_loss_total += (v_neg_elbo + v_wd_loss).item()
                valid_loss_elbo += v_neg_elbo.item() # 跟踪 ELBO
                valid_loss_wd += v_wd_loss.item()
                valid_loss_cost += v_cost.item() #  累加原始重构损失
                
            # 计算平均验证损失
            avg_valid_total = valid_loss_total / len(valid_loader)
            avg_valid_elbo = valid_loss_elbo / len(valid_loader)
            avg_valid_wd = valid_loss_wd / len(valid_loader)
            avg_valid_cost = valid_loss_cost / len(valid_loader) #  计算平均重构损失
            
            #  存储历史
            valid_history_elbo.append(avg_valid_elbo)
            valid_history_cost.append(avg_valid_cost)
            
            log_msg_valid = f"Epoch: {epoch}. Valid Loss: {avg_valid_total:.4f} (ELBO: {avg_valid_elbo:.4f}, Cost: {avg_valid_cost:.4f}, WD: {avg_valid_wd:.4f})" # (日志)
            f.write(log_msg_valid + "\n")
            logging.info(log_msg_valid)
            
            # (Fix 4b) 仅在验证损失改善时保存最佳模型
            if avg_valid_total < best_valid_loss:
                best_valid_loss = avg_valid_total
                log_msg_save = f"Epoch: {epoch}. New best validation loss: {best_valid_loss:.4f}. Saving model..."
                logging.info(log_msg_save)
                f.write(log_msg_save + "\n")
                
                model_save_path = os.path.join(model_save_dir, f"qvae_cnn_best_model_bs{BATCH_SIZE}.chkpt")
                torch.save(model.state_dict(), model_save_path)

            # --- 更新学习率调度器 ---
            scheduler.step(avg_valid_total)
            current_lr = optimizer.param_groups[0]['lr']
            log_msg_lr = f"Epoch: {epoch}. 当前学习率: {current_lr}"
            logging.info(log_msg_lr)
            f.write(log_msg_lr + "\n")

logging.info("--- QVAE 训练和评估结束 ---")

# --- 7.  绘制损失曲线 ---
def plot_losses(train_elbo, valid_elbo, train_cost, valid_cost, save_dir):
    """绘制 ELBO 和 Cost 损失曲线并保存"""
    try:
        logging.info("正在生成损失曲线图...")
        epochs_range = range(1, len(train_elbo) + 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle('QVAE 训练损失曲线', fontsize=16)

        # : 更新绘图标题
        # 子图 1: 标准 ELBO 损失
        ax1.plot(epochs_range, train_elbo, 'b-', label='训练 ELBO')
        ax1.plot(epochs_range, valid_elbo, 'r-', label='验证 ELBO')
        ax1.set_title(f'标准 ELBO 损失 (Cost + KL)')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)

        # 子图 2: 重构损失 (Cost)
        ax2.plot(epochs_range, train_cost, 'b-', label='训练重构损失 (Cost)')
        ax2.plot(epochs_range, valid_cost, 'r-', label='验证重构损失 (Cost)')
        ax2.set_title('原始重构损失 (Cost)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图像
        save_path = os.path.join(save_dir, f"loss_curves_bs{BATCH_SIZE}.png")
        plt.savefig(save_path)
        plt.close(fig) # 关闭图像
        logging.info(f"损失曲线图已保存至: {save_path}")

    except Exception as e:
        logging.error(f"绘制损失曲线时发生错误: {e}")

# 调用绘图函数
plot_losses(
    train_history_elbo, 
    valid_history_elbo, 
    train_history_cost, 
    valid_history_cost, 
    log_save_dir
)





