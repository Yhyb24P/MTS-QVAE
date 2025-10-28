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

import sys
import os
# 获取当前脚本 (generate.py) 的目录
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


# --- 导入 QVAE 相关的库 ---

from kaiwu_torch_plugin import QVAE, BoltzmannMachine, RestrictedBoltzmannMachine
from kaiwu.classical import SimulatedAnnealingOptimizer

# --- 1. 定义 QVAE 组件 (Encoder, Decoder) ---
INPUT_DIM = 1540  # 70 * 22
LATENT_DIM = 32   # 32 维隐空间

class Encoder(nn.Module):
    """编码器： p(x) -> q_logits (用于 q(z|x))"""
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, latent_dim) # 输出 q_logits
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        q_logits = self.fc3(h2)
        return q_logits

class Decoder(nn.Module):
    """解码器： z -> p_logits (用于 p(x|z))"""
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_dim) # 输出 p_logits
        self.relu = nn.ReLU()

    def forward(self, z):
        h1 = self.relu(self.fc1(z))
        h2 = self.relu(self.fc2(h1))
        p_logits = self.fc3(h2)
        return p_logits

def write_fasta(name, sequence_df):
    """
    将包含序列的 DataFrame 写入 FASTA 文件。
    DataFrame 应包含 'name' 和 'sequence' 两列。
    """
    out_file = open(name + '.fasta', "w")
    for i in range(len(sequence_df)):
        out_file.write('>' + sequence_df.name[i] + '\n') # 写入 FASTA 头部
        out_file.write(sequence_df.sequence[i] + '\n') # 写入序列
    out_file.close()

# One-hot 编码函数 (需要与 train.py 保持一致)
def one_hot_encode(seq):
    mapping = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(22)[seq2]
    
# --- 2. 初始化模型并加载权重 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载训练数据均值 (用于 train_bias)
# **重要提示**: 确保这里加载的数据集与 train.py 中使用的数据集 *完全一致*
# e.g., with open('data/mts_4_species_cleaned_train.pkl', 'rb') as f:
print("正在加载训练数据以计算均值...")
try:
    with open('data/tv_sim_split_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
except FileNotFoundError:
    print("错误: 找不到 'data/tv_sim_split_train.pkl'。")
    print("请确保 'train.py' 中使用的数据文件路径正确且文件存在。")
    sys.exit(1)

# 重新计算 One-hot 编码和均值
X_ohe_train_list = []
for i in tqdm(range(np.shape(X_train)[0]), desc="Encoding Train for mean"):
    seq = X_train.sequence[i]+'$'
    pad_seq = seq.ljust(70,'0')
    X_ohe_train_list.append(one_hot_encode(pad_seq).flatten()) # 展平为 1540
X_ohe_train = np.array(X_ohe_train_list)
mean_x = np.mean(X_ohe_train)
print(f"使用的数据集均值 (mean_x): {mean_x}")

# 初始化模型组件
rbm = RestrictedBoltzmannMachine(num_visible=LATENT_DIM, num_hidden=LATENT_DIM).to(device)
sampler = SimulatedAnnealingOptimizer(n_iterations=50, sample_size=10, trotter_number=1)
model = QVAE(Encoder(INPUT_DIM, LATENT_DIM), Decoder(LATENT_DIM, INPUT_DIM), rbm, sampler, 1.0, mean_x, LATENT_DIM).to(device)


MODEL_PATH = "model/qvae-v/qvae_mts_kl_anneal_100_batch_size_256_epochs199.chkpt"

print(f"正在从 {MODEL_PATH} 加载权重...")
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("模型权重加载成功。")
except FileNotFoundError:
    print(f"错误: 找不到模型文件 '{MODEL_PATH}'。")
    print("请确保：")
    print("1. train.py 已成功运行。")
    print("2. ANNEALING_EPOCHS (100), BATCH_SIZE (256), 和 epochs 编号 (199) 与 train.py 的输出匹配。")
    sys.exit(1)
except Exception as e:
    print(f"加载模型时出错: {e}")
    sys.exit(1)

model.eval() # 设置为评估模式

# --- 3. 从 RBM 先验分布中采样 ---
print("--- 开始从 RBM 先验分布中采样 ---")
SAMPLES = 1000 
with torch.no_grad():
    
   
    # 调整: 计算需要的迭代次数

    sampler_batch_size = 10 
    num_iterations = int(np.ceil(SAMPLES / sampler_batch_size))
    print(f"Sampler batch size = {sampler_batch_size}, 需要 {num_iterations} 次迭代以获取 {SAMPLES} 个样本。")

    z_samples_list = []
    for _ in tqdm(range(num_iterations), desc="Sampling from RBM"):
        # model.bm.sample 返回的是 (B, N) 形状的张量, B=sample_size
        z_sample_batch = model.bm.sample(sampler)
        z_samples_list.append(z_sample_batch)
    
    # 合并所有批次的样本
    z_samples = torch.cat(z_samples_list, dim=0)
    
    # 截取到精确的 SAMPLES 数量
    z_samples = z_samples[:SAMPLES]
    
    
    actual_samples = z_samples.shape[0] # 使用 z_samples 的实际行数
    print(f"已生成 {actual_samples} 个隐样本。")
    
    # 4. 使用解码器解码离散样本
    sample_logits = model.decoder(z_samples)
    
    # 5. 将 logits 转换为概率 (Sigmoid) 并移回 CPU
    sample_probs = torch.sigmoid(sample_logits).cpu()
    
    sample_probs = sample_probs.view(actual_samples, 70, 22)   

# --- 4. 解码并保存生成的序列 ---
print("解码序列中...")

# 字符到索引的映射 (同 train.py)
cdict = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))  
# 索引到字符的反向映射 (用于解码)
rev_dict = {j:i for i,j in cdict.items()}

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
        seq_to_check.append(['qvae_sample_'+str(count), sampled_seqs[i]]) # 保持 'qvae_sample_' 命名

print(f"总共生成 {np.shape(sampled_seqs)[0]} 条序列，其中 {count} 条有效（不含 '$' 或 '0'）。")

# --- 5. 保存到 FASTA 文件 ---
output_dir = "data/qvae-v/output"
os.makedirs(output_dir, exist_ok=True)

output_fasta_path = os.path.join(output_dir, "amts_kl_anneal") # 使用新文件名
seq_df = pd.DataFrame(seq_to_check, columns=['name', 'sequence'])
write_fasta(output_fasta_path, seq_df)

print(f"已将 {count} 条有效序列保存到 {output_fasta_path}.fasta")
print("--- 生成完毕 ---")

