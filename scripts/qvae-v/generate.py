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
import argparse # 导入 argparse
import torch.multiprocessing as mp 

import sys
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 获取当前脚本 (generate.py) 的目录
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 计算项目的根目录 (向上两级)
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
# 将项目根目录添加到 Python 搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logging.info(f"已将 {project_root} 添加到 sys.path")

import kaiwu as kw
kw.license.init(user_id="105879747841515522", sdk_code="4vCbDDWqIdUEXDdEHKK0L4MtOOXvMF")


# --- 导入 QVAE 相关的库 ---
from kaiwu_torch_plugin import QVAE, BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

# --- 1. 定义 QVAE 组件 (必须与 train.py 完全匹配) ---
INPUT_DIM = 1540  # 70 * 22 
LATENT_DIM = 32   # 32 维隐空间
MAX_LEN = 70      # 序列最大长度
CHANNELS = 22     # 20 个氨基酸 + '$' + '0'

# RBM (先验) 设置 (必须匹配)
prior_vis = LATENT_DIM // 2
prior_hid = LATENT_DIM - prior_vis


class Encoder(nn.Module):
    """
    编码器：使用 1D-CNN (与 train.py 匹配)
    """
    def __init__(self, input_dim, latent_dim, channels=CHANNELS, seq_len=MAX_LEN):
        super(Encoder, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) # 70 -> 35
        
        self.flat_dim = 128 * (seq_len // 2) 
        
        self.fc1 = nn.Linear(self.flat_dim, 512)
        self.fc_logits = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x): # x shape: (B, 1540)
        h = x.view(-1, self.channels, self.seq_len) # (B, 22, 70)
        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        h = self.pool(h) # (B, 128, 35)
        h = h.view(-1, self.flat_dim) # (B, 128 * 35)
        h1 = self.relu(self.fc1(h))
        return self.fc_logits(h1)

class Decoder(nn.Module):
    """
    解码器：使用 1D-CNN (与 train.py 匹配)
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
        h = self.relu(self.fc3(z.float()))
        h = self.relu(self.fc_upscale(h)) # (B, 128 * 35)
        h = h.view(-1, 128, self.start_len) # (B, 128, 35)
        h = self.unpool(h) # (B, 128, 70)
        h = self.relu(self.tconv1(h)) # (B, 64, 70)
        h = self.tconv2(h) # (B, 22, 70) -> Logits, no activation
        return h.view(-1, self.channels * self.seq_len) # (B, 1540)

# --- 必须使用 'if __name__ == "__main__":' ---
if __name__ == "__main__":

    # --- 设置 'spawn' 启动方法 ---
    try:
        mp.set_start_method('spawn', force=True) 
        logging.info("已将 multiprocessing 启动方法设置为 'spawn' 以兼容 CUDA。")
    except RuntimeError as e:
        logging.warning(f"无法设置 'spawn' 启动方法 (可能已设置): {e}")
    # --- 结束 ---

    # --- 0. 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Generate sequences from a trained QVAE model.")
    parser.add_argument("--batch_size", type=int, default=2048, 
                        help="训练时使用的 BATCH_SIZE (用于定位 .chkpt 和 mean_x.pkl 文件)。")
    parser.add_argument("--n_samples", type=int, default=5000, 
                        help="要生成的序列数量。")
    args = parser.parse_args()

    N_SAMPLES = args.n_samples
    BATCH_SIZE_TRAINED = args.batch_size


    # --- 2. 实例化模型并加载权重 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 定义模型保存目录 (与 train.py 匹配)
    model_save_dir = "model/qvae-v" 

    # --- 加载 mean_x ---
    mean_x_load_path = os.path.join(model_save_dir, f"mean_x_bs{BATCH_SIZE_TRAINED}.pkl")
    
    try:
        with open(mean_x_load_path, 'rb') as f:
            mean_x_value = pickle.load(f)
        logging.info(f"成功从 {mean_x_load_path} 加载 train_bias (mean_x): {mean_x_value}")
    except FileNotFoundError:
        logging.error(f"错误: 找不到 {mean_x_load_path}。")
        logging.error(f"请确保您已运行 train.py (使用 BATCH_SIZE={BATCH_SIZE_TRAINED}) 来生成此文件。")
        sys.exit(1)
    except Exception as e:
        logging.error(f"加载 mean_x 失败: {e}")
        sys.exit(1)
    # --- 结束 ---


    # 1. 实例化 CNN 组件
    encoder = Encoder(INPUT_DIM, LATENT_DIM, channels=CHANNELS, seq_len=MAX_LEN).to(device)
    decoder = Decoder(LATENT_DIM, INPUT_DIM, channels=CHANNELS, seq_len=MAX_LEN).to(device)

    # 2.  实例化玻尔兹曼机先验 (RBM)
    bm_prior = RestrictedBoltzmannMachine(
        num_visible=prior_vis, # 16
        num_hidden=prior_hid   # 16
    ).to(device)
    logging.info(f"已初始化 RestrictedBoltzmannMachine 先验，总共 {bm_prior.num_nodes} 个节点。")

    # 3. 实例化强探索采样器
    logging.info("配置强探索采样器 (SimulatedAnnealingOptimizer)...")
    sampler = SimulatedAnnealingOptimizer(
        initial_temperature=500.0,
        alpha=0.999,
        cutoff_temperature=0.001,
        iterations_per_t=100,
        size_limit=N_SAMPLES,         # 在此处指定样本数量
        process_num = -1              # 使用 'spawn' 时可以安全使用多进程
    )
    logging.info(f"采样器进程数设置为: {sampler.process_num} (已启用 'spawn' 模式)")

    # 4. 实例化 QVAE 主模型
    model = QVAE(
        encoder=encoder,
        decoder=decoder,
        bm=bm_prior,
        sampler=sampler,
        dist_beta=1.0,           
        mean_x=mean_x_value, # 传递加载的 float 类型的 mean_x
        num_vis=bm_prior.num_visible 
    ).to(device)

    # 5. 加载训练好的 'best' 权重
    model_load_path = os.path.join(model_save_dir, f"qvae_cnn_best_model_bs{BATCH_SIZE_TRAINED}.chkpt")
    logging.info(f"正在从 {model_load_path} 加载模型权重...")

    try:
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        logging.info("模型权重加载成功。")
    except FileNotFoundError:
        logging.error(f"错误: 找不到模型文件 {model_load_path}")
        logging.error("请确保 --batch_size 参数与您训练保存的文件名匹配。")
        sys.exit(1) # 退出脚本

    model.eval() # 切换到评估模式

    # --- 3. 从先验 (RBM) 中采样隐变量 ---

    # 定义解码字典
    rev_dict = dict(zip(range(22), "FIWLVMYCATHGSQRKNEPD$0"))

    with torch.no_grad():
        logging.info(f"正在从 RBM 先验中采样 {N_SAMPLES} 个隐变量...")
        
        try:
            # RBM 基类有 'sample' 方法，该方法接收一个采样器
            z_samples = model.bm.sample(model.sampler)
            logging.info("采样完成。")
        except Exception as e:
            logging.error(f"调用 model.bm.sample(model.sampler) 时出错: {e}")
            logging.exception("详细错误信息:") # 打印完整的堆栈跟踪
            sys.exit(1)
        
        z_samples = z_samples.to(device) # 确保样本在正确的设备上
        
        actual_samples = z_samples.shape[0] 
        logging.info(f"已生成 {actual_samples} 个隐样本。")
        
        # 4. 使用解码器解码离散样本
        sample_logits = model.decoder(z_samples)
        
        # 5. 将 logits 转换为概率 (Sigmoid) 并移回 CPU
        sample_probs = torch.sigmoid(sample_logits).cpu()
        
        # 确保 reshape 维度与 CNN 输出匹配
        sample_probs = sample_probs.view(actual_samples, MAX_LEN, CHANNELS) # (B, 70, 22)   

    # --- 4. 解码并保存生成的序列 ---
    logging.info("解码序列中...")
    sampled_seqs = []
    for i, seq_probs in enumerate(tqdm(sample_probs, desc="解码")):
        out_seq = []
        for j, pos_probs in enumerate(seq_probs):
            best_idx = pos_probs.argmax()
            out_seq.append(rev_dict[best_idx.item()])
            
        final_seq = ''.join(out_seq).rstrip('0').rstrip('$')
        sampled_seqs.append(final_seq) 

    seq_to_check = []
    count = 0
    for i in range(np.shape(sampled_seqs)[0]):
        # 确保序列不为空且不包含填充/停止符
        if sampled_seqs[i] and '$' not in sampled_seqs[i] and '0' not in sampled_seqs[i]:
            count = count + 1
            seq_to_check.append(['qvae_sample_'+str(count), sampled_seqs[i]])

    logging.info(f"总共生成的有效序列: {len(seq_to_check)}")
    filtered_seq_to_check = pd.DataFrame(seq_to_check, columns = ['name', 'sequence'])
        
    logging.info(f'序列总数 (去重前): {len(filtered_seq_to_check)}')
    filtered_seq_to_check = filtered_seq_to_check.drop_duplicates(subset='sequence').reset_index().drop('index', axis = 1)
    logging.info(f'序列总数 (去重后): {len(filtered_seq_to_check)}')

    # --- 5. 保存到 FASTA 文件 ---
    output_dir = "data/qvae-v/output" 
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"generated_seqs_best_b{BATCH_SIZE_TRAINED}_n{N_SAMPLES}.fasta")

    with open(output_filename, 'w') as f:
        for index, row in filtered_seq_to_check.iterrows():
            f.write(">" + row['name'] + "\n")
            f.write(row['sequence'] + "\n")

    logging.info(f"已将 {len(filtered_seq_to_check)} 条生成的序列保存到: {output_filename}")
