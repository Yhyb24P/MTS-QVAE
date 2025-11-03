import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch
# 移除了 WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import logging
import itertools
# 移除了 FASTA 和 SKLearn 相关的导入

import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logging.info(f"已将 {project_root} 添加到 sys.path")


import kaiwu as kw
kw.license.init(user_id="105879747841515522", sdk_code="4vCbDDWqIdUEXDdEHKK0L4MtOOXvMF")

from kaiwu_torch_plugin import QVAE, BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

# --- 1. 设置与常量 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {device}")

INPUT_DIM = 1540
LATENT_DIM = 64
BATCH_SIZE = 2048
LEARNING_RATE_VAE = 1e-4
LEARNING_RATE_BM = 1e-5
EPOCHS = 50
NUM_WORKERS = 4
MAX_LEN = 70
CHANNELS = 22

# -----------------------------------------------------------------
# --- 优化点: 引入 BETA (β) 参数来平衡保真度和多样性 ---
# β = 1.0 时, 模式坍塌 (低多样性)
# β < 1.0 时, 多样性会提高
BETA = 0.1
# -----------------------------------------------------------------

prior_vis = LATENT_DIM // 2
prior_hid = LATENT_DIM - prior_vis

log_save_dir = "data/qvae-v/model"
model_save_dir = "model/qvae-v"
os.makedirs(log_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# 更新文件名以包含 BETA 值，防止覆盖
log_file_path = os.path.join(log_save_dir, f"loss_qvae_cnn_bs{BATCH_SIZE}_ld{LATENT_DIM}_beta{BETA}.txt")

# --- 2. 数据加载与预处理 (恢复 .pkl 加载逻辑) ---
MAPPING = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(CHANNELS)))

def one_hot_encode(seq):
    seq2 = [MAPPING[i] for i in seq]
    return np.eye(CHANNELS)[seq2]

class SequenceDataset(Dataset):
    def __init__(self, pkl_file_path, max_len):
        logging.info(f"Loading data from {pkl_file_path}...")
        try:
            with open(pkl_file_path, 'rb') as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            logging.error(f"错误: 找不到 .pkl 文件: {pkl_file_path}")
            sys.exit(1)
            
        self.max_len = max_len
        self.pad_char = '0'
        self.end_char = '$'
        logging.info(f"Loaded {len(self.data)} sequences.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_raw = self.data.sequence[idx]
        seq_terminated = (seq_raw + self.end_char)[:self.max_len]
        seq_padded = seq_terminated.ljust(self.max_len, self.pad_char)
        ohe_seq = one_hot_encode(seq_padded)
        ohe_tensor_flat = torch.FloatTensor(ohe_seq).view(-1)
        # 仅返回数据
        return ohe_tensor_flat

def get_mean_x(dataset, batch_size, num_workers):
    logging.info("Calculating mean_x from training data...")
    temp_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    total_sum = 0.0
    total_count = 0
    for batch_data in tqdm(temp_loader, desc="Calculating mean_x"):
        total_sum += batch_data.sum().item()
        total_count += batch_data.numel()
    mean_x = total_sum / total_count
    logging.info(f"Calculated train_bias (mean_x): {mean_x}")
    return mean_x


# --- 4. 定义 QVAE 模型组件 ---
class Encoder(nn.Module):
# ... (与之前 LD=64 版本相同) ...
    def __init__(self, input_dim, latent_dim, channels=CHANNELS, seq_len=MAX_LEN):
        super(Encoder, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flat_dim = 128 * (seq_len // 2)
        self.fc1 = nn.Linear(self.flat_dim, 512)
        self.fc_logits = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = x.view(-1, self.channels, self.seq_len)
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        h = self.pool(h)
        h = h.view(-1, self.flat_dim)
        h1 = self.relu(self.fc1(h))
        return self.fc_logits(h1)

class Decoder(nn.Module):
# ... (与之前 LD=64 版本相同) ...
    def __init__(self, latent_dim, output_dim, channels=CHANNELS, seq_len=MAX_LEN):
        super(Decoder, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.start_len = seq_len // 2
        self.start_flat_dim = 128 * self.start_len
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc_upscale = nn.Linear(512, self.start_flat_dim)
        self.unpool = nn.Upsample(scale_factor=2)
        self.tconv1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.tconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, z):
        h = self.relu(self.fc3(z.float()))
        h = self.relu(self.fc_upscale(h))
        h = h.view(-1, 128, self.start_len)
        h = self.unpool(h)
        h = self.relu(self.tconv1(h))
        h = self.tconv2(h)
        return h.view(-1, self.channels * self.seq_len)

def plot_losses(train_elbo, valid_elbo, train_cost, valid_cost, train_bm, save_dir):
# ... (函数保持不变) ...
    try:
        logging.info("正在生成损失曲线图...")
        epochs_range = range(1, len(train_elbo) + 1)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))
        fig.suptitle(f'QVAE Training Loss Curves (LD={LATENT_DIM}, β={BETA})', fontsize=16)
        ax1.plot(epochs_range, train_elbo, 'b-', label='Train ELBO')
        ax1.plot(epochs_range, valid_elbo, 'r-', label='Valid ELBO')
        ax1.set_title(f'ELBO Loss (Cost + {BETA}*KL_lite)')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(epochs_range, train_cost, 'b-', label='Train Reconstruction Loss (Cost)')
        ax2.plot(epochs_range, valid_cost, 'r-', label='Valid Reconstruction Loss (Cost)')
        ax2.set_title('Reconstruction Loss (Cost)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        ax3.plot(epochs_range, train_bm, 'g-', label='Train BM CD Loss')
        ax3.set_title('BM Contrastive Divergence Loss (E_q[E] - E_p[E])')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f"loss_curves_bs{BATCH_SIZE}_ld{LATENT_DIM}_beta{BETA}.png")
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"损失曲线图已保存至: {save_path}")
    except Exception as e:
        logging.error(f"绘制损失曲线时发生错误: {e}")

def plot_rbm_weights(weights_matrix, save_dir, epoch):
# ... (函数保持不变) ...
    try:
        logging.info(f"正在生成 Epoch {epoch} 的 RBM 权重热力图...")
        plt.figure(figsize=(12, 12))
        sns.heatmap(weights_matrix, cmap='viridis', square=True, annot=False)
        plt.title(f'RBM Weight Matrix (W) - Epoch {epoch} - LD {LATENT_DIM} - beta {BETA}')
        plt.xlabel('Nodes')
        plt.ylabel('Nodes')
        save_path = os.path.join(save_dir, f"rbm_weights_heatmap_epoch_{epoch}_bs{BATCH_SIZE}_ld{LATENT_DIM}_beta{BETA}.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"RBM 权重热力图已保存至: {save_path}")
    except Exception as e:
        logging.error(f"绘制 RBM 权重热力图时发生错误: {e}")


def main():
    # --- 3. 实例化数据集和加载器 (已恢复) ---
    train_dataset = SequenceDataset(pkl_file_path='data/tv_sim_split_train.pkl', max_len=MAX_LEN)
    valid_dataset = SequenceDataset(pkl_file_path='data/tv_sim_split_valid.pkl', max_len=MAX_LEN)
    
    mean_x = get_mean_x(train_dataset, BATCH_SIZE, NUM_WORKERS)
    
    # 更新文件名以包含 BETA
    mean_x_save_path = os.path.join(model_save_dir, f"mean_x_bs{BATCH_SIZE}_ld{LATENT_DIM}_beta{BETA}.pkl")
    with open(mean_x_save_path, 'wb') as f:
        pickle.dump(mean_x, f)
    logging.info(f"已将 mean_x 保存到: {mean_x_save_path}")

    logging.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(valid_dataset)}")

    # --- 恢复标准加载器 (无加权采样) ---
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, # 恢复 shuffle
        num_workers=NUM_WORKERS,
        pin_memory=True,
        sampler=None # 移除 sampler
    )
    
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --- 5. 实例化模型和优化器 ---
    encoder = Encoder(INPUT_DIM, LATENT_DIM).to(device)
    decoder = Decoder(LATENT_DIM, INPUT_DIM).to(device)
    bm_prior = RestrictedBoltzmannMachine(
        num_visible=prior_vis,
        num_hidden=prior_hid
    ).to(device)
    logging.info(f"已初始化 RestrictedBoltzmannMachine 先验，总共 {bm_prior.num_nodes} 个节点。")
    logging.info("配置训练采样器 (SimulatedAnnealingOptimizer)...")
    
    train_sampler = SimulatedAnnealingOptimizer(
        initial_temperature=500.0,
        alpha=0.99,
        cutoff_temperature=0.001,
        iterations_per_t=20,
        size_limit=100,
        process_num=-1
    )
    
    model = QVAE(
        encoder=encoder,
        decoder=decoder,
        bm=bm_prior,
        sampler=train_sampler,
        dist_beta=1.0,
        mean_x=mean_x,
        num_vis=bm_prior.num_visible
    ).to(device)

    vae_params = itertools.chain(model.encoder.parameters(), model.decoder.parameters())
    bm_params = model.bm.parameters()
    optimizer_vae = optim.Adam(vae_params, lr=LEARNING_RATE_VAE)
    optimizer_bm = optim.Adam(bm_params, lr=LEARNING_RATE_BM)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_vae, mode='min', factor=0.2, patience=3, min_lr=1e-6)

    # --- 6. 训练循环 ---
    logging.info(f"--- 开始训练 (使用 .pkl, β-VAE 方案): {EPOCHS} 个 Epochs, β={BETA} ---")
    best_valid_loss = float('inf')

    train_history_elbo = []
    valid_history_elbo = []
    train_history_cost = []
    valid_history_cost = []
    train_history_bm_loss = []

    try: 
        with open(log_file_path, 'w') as f:
            f.write(f"使用双重梯度流 (VAE Loss + BM Loss) (无加权采样, β={BETA})\n")
            f.write(f"VAE LR: {LEARNING_RATE_VAE}, BM LR: {LEARNING_RATE_BM}, LATENT_DIM: {LATENT_DIM}\n")

            for epoch in range(EPOCHS):
                model.train()
                train_loss_total = 0.0
                train_loss_elbo = 0.0
                train_loss_wd = 0.0
                train_loss_cost = 0.0
                train_loss_bm = 0.0
                
                for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
                    batch_X = batch_data.to(device) 
                    
                    optimizer_vae.zero_grad()
                    optimizer_bm.zero_grad()
                    
                    # -----------------------------------------------------------------
                    # --- 优化点: 传入 BETA 参数 ---
                    (output, recon_x, neg_elbo, wd_loss,
                     total_kl, cost, q, posterior, zeta) = model.neg_elbo(batch_X, kl_beta=BETA) # <-- 在这里使用 BETA
                    # -----------------------------------------------------------------
                    
                    vae_loss = neg_elbo + wd_loss
                    vae_loss.backward(retain_graph=True) 
                    bm_loss = model.get_bm_loss(posterior, zeta)
                    bm_loss.backward()
                    optimizer_vae.step()
                    optimizer_bm.step()
                    train_loss_total += vae_loss.item()
                    train_loss_elbo += neg_elbo.item()
                    train_loss_wd += wd_loss.item()
                    train_loss_cost += cost.item()
                    train_loss_bm += bm_loss.item()

                avg_train_total = train_loss_total / len(train_loader)
                avg_train_elbo = train_loss_elbo / len(train_loader)
                avg_train_wd = train_loss_wd / len(train_loader)
                avg_train_cost = train_loss_cost / len(train_loader)
                avg_train_bm_loss = train_loss_bm / len(train_loader)
                train_history_elbo.append(avg_train_elbo)
                train_history_cost.append(avg_train_cost)
                train_history_bm_loss.append(avg_train_bm_loss)
                log_msg_train = (
                    f"Epoch: {epoch}. Train VAE Loss: {avg_train_total:.4f} "
                    f"(ELBO: {avg_train_elbo:.4f}, Cost: {avg_train_cost:.4f}, WD: {avg_train_wd:.4f}) | "
                    f"Train BM Loss: {avg_train_bm_loss:.4f}"
                )
                f.write(log_msg_train + "\n")
                logging.info(log_msg_train)

                model.eval()
                with torch.no_grad():
                    valid_loss_total = 0.0
                    valid_loss_elbo = 0.0
                    valid_loss_wd = 0.0
                    valid_loss_cost = 0.0
                    valid_loss_bm = 0.0
                    for batchv_data in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]"):
                        batchv_X = batchv_data.to(device)
                        
                        # -----------------------------------------------------------------
                        # --- 优化点: 传入 BETA 参数 ---
                        (v_output, v_recon_x, v_neg_elbo, v_wd_loss,
                         v_total_kl, v_cost, v_q, v_posterior, v_zeta) = model.neg_elbo(batchv_X, kl_beta=BETA) # <-- 在这里使用 BETA
                        # -----------------------------------------------------------------
                        
                        v_bm_loss = model.get_bm_loss(v_posterior, v_zeta)
                        valid_loss_total += (v_neg_elbo + v_wd_loss).item()
                        valid_loss_elbo += v_neg_elbo.item()
                        valid_loss_wd += v_wd_loss.item()
                        valid_loss_cost += v_cost.item()
                        valid_loss_bm += v_bm_loss.item()
                        
                    avg_valid_total = valid_loss_total / len(valid_loader)
                    avg_valid_elbo = valid_loss_elbo / len(valid_loader)
                    avg_valid_wd = valid_loss_wd / len(valid_loader)
                    avg_valid_cost = valid_loss_cost / len(valid_loader)
                    avg_valid_bm_loss = valid_loss_bm / len(valid_loader)
                    valid_history_elbo.append(avg_valid_elbo)
                    valid_history_cost.append(avg_valid_cost)
                    log_msg_valid = (
                        f"Epoch: {epoch}. Valid VAE Loss: {avg_valid_total:.4f} "
                        f"(ELBO: {avg_valid_elbo:.4f}, Cost: {avg_valid_cost:.4f}, WD: {avg_valid_wd:.4f}) | "
                        f"Valid BM Loss: {avg_valid_bm_loss:.4f}"
                    )
                    f.write(log_msg_valid + "\n")
                    logging.info(log_msg_valid)

                    if avg_valid_total < best_valid_loss:
                        best_valid_loss = avg_valid_total
                        # 更新文件名以包含 BETA
                        save_path = os.path.join(model_save_dir, f"qvae_cnn_best_model_bs{BATCH_SIZE}_ld{LATENT_DIM}_beta{BETA}.chkpt")
                        torch.save(model.state_dict(), save_path)
                        logging.info(f"Epoch: {epoch}. New best validation VAE loss: {best_valid_loss:.4f}. Saving model to {save_path}...")

                    current_lr_vae = optimizer_vae.param_groups[0]['lr']
                    current_lr_bm = optimizer_bm.param_groups[0]['lr']
                    log_msg_lr = f"Epoch: {epoch}. 当前学习率 (VAE): {current_lr_vae}, (BM): {current_lr_bm}"
                    f.write(log_msg_lr + "\n")
                    logging.info(log_msg_lr)
                    try:
                        with torch.no_grad():
                            q_coef = model.bm.quadratic_coef.detach().cpu().numpy()
                            l_bias = model.bm.linear_bias.detach().cpu().numpy()
                            q_stats = f"BM W (二次项): Mean={q_coef.mean():.4f}, Std={q_coef.std():.4f}, Min={q_coef.min():.4f}, Max={q_coef.max():.4f}"
                            l_stats = f"BM h (一次项): Mean={l_bias.mean():.4f}, Std={l_bias.std():.4f}, Min={l_bias.min():.4f}, Max={l_bias.max():.4f}"
                            logging.info(q_stats)
                            logging.info(l_stats)
                            f.write(q_stats + "\n")
                            f.write(l_stats + "\n")
                    except Exception as e:
                        logging.error(f"无法获取 RBM 参数统计信息: {e}")
                scheduler.step(avg_valid_elbo)

    except KeyboardInterrupt:
        logging.warning("--- 训练被用户中断 (KeyboardInterrupt) ---")
    finally:
        logging.info("--- QVAE 训练和评估结束 ---")
        if train_history_elbo:
            plot_losses(
                train_history_elbo,
                valid_history_elbo,
                train_history_cost,
                valid_history_cost,
                train_history_bm_loss,
                log_save_dir
            )
        else:
            logging.info("没有足够的训练数据来绘制损失曲线。")
        try:
            logging.info("正在获取最终的 RBM 权重...")
            with torch.no_grad():
                final_weights = model.bm.quadratic_coef.detach().cpu().numpy()
            plot_rbm_weights(final_weights, log_save_dir, 'final')
        except Exception as e:
            logging.error(f"无法绘制 RBM 权重热力图: {e}")

if __name__ == "__main__":
    main()

