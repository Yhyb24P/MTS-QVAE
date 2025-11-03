import pandas as pd
import numpy as np
import pickle 
from tqdm import tqdm
import re
from Bio import SeqIO
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
import torch
import argparse
import torch.multiprocessing as mp 

import sys
import os
import logging
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

# --- 1. 定义 QVAE 组件 (必须与 train_optimized.py 完全匹配) ---

# -----------------------------------------------------------------
# --- 优化点 1: 匹配新的隐空间维度 ---
LATENT_DIM = 64   # 必须匹配
BETA = 0.1        # 必须匹配
BATCH_SIZE = 2048 # 必须匹配
# -----------------------------------------------------------------

INPUT_DIM = 1540  
MAX_LEN = 70      
CHANNELS = 22     
N_SAMPLES = 5000  # 您希望生成的样本数量

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


# --- 2. 主执行函数 ---
def main():
    try:
        mp.set_start_method('spawn')
        logging.info("已将 multiprocessing 启动方法设置为 'spawn' 以兼容 CUDA。")
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            logging.warning(f"无法设置 'spawn' 启动方法: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # -----------------------------------------------------------------
    # --- 优化点 2: 加载新的 beta=0.1 文件 ---
    model_save_dir = "model/qvae-v"
    mean_x_path = os.path.join(model_save_dir, f"mean_x_bs{BATCH_SIZE}_ld{LATENT_DIM}_beta{BETA}.pkl")
    model_path = os.path.join(model_save_dir, f"qvae_cnn_best_model_bs{BATCH_SIZE}_ld{LATENT_DIM}_beta{BETA}.chkpt")
    # -----------------------------------------------------------------

    try:
        with open(mean_x_path, 'rb') as f:
            mean_x = pickle.load(f)
        logging.info(f"成功从 {mean_x_path} 加载 train_bias (mean_x): {mean_x}")
    except FileNotFoundError:
        logging.error(f"错误: 找不到 {mean_x_path}。")
        logging.error(f"请确保您已运行 train.py 且 BS, LD 和 BETA 参数匹配。")
        return
        
    prior_vis = LATENT_DIM // 2
    prior_hid = LATENT_DIM - prior_vis

    bm_prior = RestrictedBoltzmannMachine(
        num_visible=prior_vis,
        num_hidden=prior_hid
    ).to(device)
    logging.info(f"已初始化 RestrictedBoltzmannMachine 先验，总共 {bm_prior.num_nodes} 个节点。")

    logging.info("配置强探索采样器 (SimulatedAnnealingOptimizer)...")
    # 保持高探索性设置以最大化多样性
    sampler = SimulatedAnnealingOptimizer(
        initial_temperature=5000.0,
        alpha=0.995,
        cutoff_temperature=0.001,
        iterations_per_t=100,
        size_limit=N_SAMPLES,
        process_num = -1
    )
    
    if sampler.process_num > 0:
        logging.info(f"采样器进程数设置为: {sampler.process_num} (已启用 'spawn' 模式)")
    else:
        logging.info(f"采样器进程数设置为: 自动 (CPU 核心数)")

    model = QVAE(
        encoder=Encoder(INPUT_DIM, LATENT_DIM),
        decoder=Decoder(LATENT_DIM, INPUT_DIM),
        bm=bm_prior,
        sampler=sampler,
        dist_beta=1.0,
        mean_x=mean_x,
        num_vis=bm_prior.num_visible
    ).to(device)

    logging.info(f"正在从 {model_path} 加载模型权重...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("模型权重加载成功。")
    except FileNotFoundError:
        logging.error(f"错误: 找不到模型文件 {model_path}。")
        return
    except Exception as e:
        logging.error(f"加载模型权重时出错: {e}")
        return
        
    model.eval()

    # --- 3. 从 RBM 先验中采样 ---
    logging.info(f"正在从 RBM 先验中采样 {N_SAMPLES} 个隐变量 (使用高探索性设置)...")
    z_samples_list = []
    # (如果N_SAMPLES很大，可能需要分批)
    z_samples_list = model.bm.sample(model.sampler)
    z_samples_tensor = z_samples_list.to(device)
    logging.info("采样完成。")
    logging.info(f"已生成 {len(z_samples_tensor)} 个隐样本。")

    # --- 4. 解码 & 保存 ---
    cdict = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(CHANNELS)))
    rev_dict = {j:i for i,j in cdict.items()}
    
    logging.info("解码序列中...")
    all_seq_probs = []
    
    # 为防止内存溢出，分批解码
    decode_batch_size = 512 
    with torch.no_grad():
        for i in tqdm(range(0, len(z_samples_tensor), decode_batch_size), desc="解码"):
            batch_z = z_samples_tensor[i:i+decode_batch_size]
            recon_x_logits = model.decoder(batch_z) # (B, 1540)
            seq_probs = recon_x_logits.view(-1, MAX_LEN, CHANNELS).cpu() # (B, 70, 22)
            all_seq_probs.append(seq_probs)
            
    sample_probs = torch.cat(all_seq_probs, dim=0)
    
    sampled_seqs = []
    for i, seq_probs in enumerate(tqdm(sample_probs, desc="处理序列")):
        out_seq = []
        for j, pos_probs in enumerate(seq_probs):
            best_idx = pos_probs.argmax()
            out_seq.append(rev_dict[best_idx.item()])
        final_seq = ''.join(out_seq).rstrip('0').rstrip('$')
        sampled_seqs.append(final_seq) 

    seq_to_check = []
    count = 0
    for i in range(np.shape(sampled_seqs)[0]):
        if sampled_seqs[i] and '$' not in sampled_seqs[i] and '0' not in sampled_seqs[i]:
            count = count + 1
            seq_to_check.append(['qvae_sample_'+str(count), sampled_seqs[i]])

    logging.info(f"总共生成的有效序列: {len(seq_to_check)}")
    filtered_seq_to_check = pd.DataFrame(seq_to_check, columns = ['name', 'sequence'])
        
    logging.info(f'序列总数 (去重前): {len(filtered_seq_to_check)}')
    filtered_seq_to_check = filtered_seq_to_check.drop_duplicates(subset='sequence').reset_index().drop('index', axis = 1)
    logging.info(f'序列总数 (去重后): {len(filtered_seq_to_check)}')

    # -----------------------------------------------------------------
    # --- 优化点 3: 保存到新的 beta=0.1 文件 ---
    output_dir = "data/qvae-v/output" 
    os.makedirs(output_dir, exist_ok=True)
    output_fasta_path = os.path.join(output_dir, f"generated_seqs_best_b{BATCH_SIZE}_ld{LATENT_DIM}_beta{BETA}_n{N_SAMPLES}.fasta")
    # -----------------------------------------------------------------
    
    try:
        with open(output_fasta_path, "w") as out_file:
            for index, row in filtered_seq_to_check.iterrows():
                out_file.write(f">{row['name']}\n")
                out_file.write(f"{row['sequence']}\n")
        logging.info(f"已将 {len(filtered_seq_to_check)} 条生成的序列保存到: {output_fasta_path}")
    except Exception as e:
        logging.error(f"保存 FASTA 文件时出错: {e}")

if __name__ == "__main__":
    main()