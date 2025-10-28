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


# --- TODO: 导入 QVAE 相关的库 ---

from kaiwu_torch_plugin import QVAE, BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

# --- 0. 添加命令行参数解析 ---
parser = argparse.ArgumentParser(description="Generate sequences from a trained QVAE model.")
parser.add_argument("--epoch", type=int, required=True, help="要加载的 checkpoint 的 Epoch 编号。")
parser.add_argument("--batch_size", type=int, default=2048, help="训练时使用的 BATCH_SIZE (用于查找正确的文件)。")
parser.add_argument("--n_samples", type=int, default=1000, help="要生成的序列数量。")
args = parser.parse_args()


# --- 1. 定义 QVAE 组件 ---
INPUT_DIM = 1540  # 70 * 22
LATENT_DIM = 32   # 32 维隐空间

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

# 辅助函数 
def write_fasta(name, sequence_df):
    out_file = open(name + '.fasta', "w")
    for i in range(len(sequence_df)):
        out_file.write('>' + sequence_df.name[i] + '\n')
        out_file.write(sequence_df.sequence[i] + '\n')
    out_file.close()

# --- 2. 实例化模型并加载权重 ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {device}")

# 1. 实例化所有组件
encoder = Encoder(INPUT_DIM, LATENT_DIM).to(device)
decoder = Decoder(LATENT_DIM, INPUT_DIM).to(device)

# 2.  实例化玻尔兹曼机先验 (RBM)
# 必须与 train.py 完全一致
prior_vis = LATENT_DIM // 2
prior_hid = LATENT_DIM - prior_vis
bm_prior = RestrictedBoltzmannMachine(
    num_visible=prior_vis, # 16
    num_hidden=prior_hid   # 16
).to(device)
logging.info(f"已初始化 RestrictedBoltzmannMachine 先验，总共 {bm_prior.num_nodes} 个节点 ({bm_prior.num_visible} 可见 + {bm_prior.num_hidden} 隐藏)。")


# 3. 实例化采样器 (与 train_modified.py 保持一致)
logging.info("正在配置 SimulatedAnnealingOptimizer (与训练时参数一致)...")
sampler = SimulatedAnnealingOptimizer(
    initial_temperature=500,  # 提高初始温度以进行更广泛的探索
    alpha=0.999,              # 保持慢速降温
    iterations_per_t=100      # 增加每个温度的迭代深度
)

# 4. 实例化 QVAE 主模型
# mean_x 是在整个训练集上计算的，与 batch_size 无关，保持不变
mean_x_value = 0.04545454680919647 
logging.info(f"使用 mean_x: {mean_x_value}")
model = QVAE(
    encoder=encoder,
    decoder=decoder,
    bm=bm_prior,
    sampler=sampler,
    dist_beta=1.0,           
    mean_x=mean_x_value, 
    num_vis=bm_prior.num_visible 
).to(device)

# 5. 根据命令行参数加载训练好的权重
model_path = f"model/qvae/qvae_mts_kl_weight_1_batch_size_{args.batch_size}_epochs{args.epoch}.chkpt"
logging.info(f"从以下路径加载模型: {model_path}")

if not os.path.exists(model_path):
    logging.info(f"错误: 找不到模型文件 {model_path}")
    logging.info("请确保 --epoch 和 --batch_size 参数与您训练保存的文件名一致。")
    sys.exit(1) # 退出脚本

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # 设置为评估模式

# 字典
cdict = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))  
rev_dict = {j:i for i,j in cdict.items()}

N_SAMPLES_TO_GENERATE = args.n_samples

# --- 3.从 QVAE 的玻尔兹曼机先验 p(z) 采样 ---

logging.info(f"从 QVAE 先验 (RBM) 采样 {N_SAMPLES_TO_GENERATE} 个序列...")
with torch.no_grad():
    
    # 1. 获取训练好的玻尔兹曼机 (先验)
    trained_prior = model.bm.to(device) 
    
    # 2. 配置采样器以生成 N_SAMPLES_TO_GENERATE 个样本
    sampler.size_limit = N_SAMPLES_TO_GENERATE
    
    # 3. sample() 接口
    logging.info("运行采样器以从先验获取 z_samples... (这可能需要一些时间)")
    z_samples = trained_prior.sample(sampler) # 返回 (N_SAMPLES, 32)
    z_samples = z_samples.to(device)
    
    actual_samples = z_samples.shape[0] # 使用 z_samples 的实际行数
    logging.info(f"已生成 {actual_samples} 个隐样本。")
    
    # 4. 使用解码器解码离散样本
    sample_logits = model.decoder(z_samples)
    
    # 5. 将 logits 转换为概率 (Sigmoid) 并移回 CPU
    sample_probs = torch.sigmoid(sample_logits).cpu()
    
    sample_probs = sample_probs.view(actual_samples, 70, 22)   

# --- 4. 解码并保存生成的序列 ---
logging.info("解码序列中...")
sampled_seqs = []
for i, seq_probs in enumerate(sample_probs):
    out_seq = []
    for j, pos_probs in enumerate(seq_probs):
        best_idx = pos_probs.argmax()
        out_seq.append(rev_dict[best_idx.item()])
        
    final_seq = ''.join(out_seq).rstrip('0').rstrip('$')
    sampled_seqs.append(final_seq) 

seq_to_check = []
count = 0
for i in range(np.shape(sampled_seqs)[0]):
    if sampled_seqs[i].find('$') == - 1 and sampled_seqs[i].find('0') == - 1:
        count = count + 1
        seq_to_check.append(['qvae_sample_'+str(count), sampled_seqs[i]])

logging.info(f"总共生成的有效序列: {len(seq_to_check)}")
filtered_seq_to_check = pd.DataFrame(seq_to_check, columns = ['name', 'sequence'])
    
logging.info('序列总数:', len(filtered_seq_to_check))
filtered_seq_to_check = filtered_seq_to_check.drop_duplicates(subset='sequence').reset_index().drop('index', axis=1)
logging.info('去重后剩余序列总数', len(filtered_seq_to_check))

# --- 确保输出目录存在 ---
output_dir = "data/qvae-v/output"
os.makedirs(output_dir, exist_ok=True)
logging.info(f"确保输出目录存在: {output_dir}")

output_fasta_name = os.path.join(output_dir, f'amts_bs{args.batch_size}_epoch{args.epoch}_n{len(filtered_seq_to_check)}')
logging.info(f"正在将最终序列写入 {output_fasta_name}.fasta")
write_fasta(output_fasta_name, filtered_seq_to_check)

logging.info("生成完毕。")
