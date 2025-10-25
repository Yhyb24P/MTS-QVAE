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

# --- TODO: 导入 QVAE 相关的库 ---
# 导入的类必须与 train.py 中的定义完全一致
try:
    from kaiwu_torch_plugin import QVAE, RestrictedBoltzmannMachine
    from kaiwu_torch_plugin.qvae_dist_util import FactorialBernoulliUtil
    from kaiwu_torch_plugin.samplers import SimulatedAnnealingSampler 
except ImportError:
    print("="*50)
    print("错误：无法导入 QVAE 库 (kaiwu_torch_plugin)。")
    print("请确保导入的类与 train.py 中定义的一致。")
    print("="*50)
    raise

# --- 1. 定义 QVAE 组件 (必须与 train.py 一致) ---
# 为了加载模型，我们必须先创建完全相同的架构实例

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

# 辅助函数 (来自原 generate.py)
def write_fasta(name, sequence_df):
    out_file = open(name + '.fasta', "w")
    for i in range(len(sequence_df)):
        out_file.write('>' + sequence_df.name[i] + '\n')
        out_file.write(sequence_df.sequence[i] + '\n')
    out_file.close()

# --- 2. 实例化模型并加载权重 ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 实例化所有组件
encoder = Encoder(INPUT_DIM, LATENT_DIM)
decoder = Decoder(LATENT_DIM, INPUT_DIM)
rbm_prior = RestrictedBoltzmannMachine(n_visible=LATENT_DIM, n_hidden=64) # 必须与训练时相同

# 2. 实例化采样器 (TODO: 必须提供)
# 这里我们需要采样器来从先验 p(z) 中采样
sampler = SimulatedAnnealingSampler(n_steps=100) # 确保配置与训练时兼容

# 3. 实例化 QVAE 主模型
model = QVAE(
    encoder=encoder,
    decoder=decoder,
    bm=rbm_prior,
    recon_dist_util=FactorialBernoulliUtil,
    sampler=sampler
).to(device)

# 4. 加载训练好的权重
# TODO: 确保这里的路径和 epoch 与您最终训练好的模型一致
model_path = "model/qvae_mts_kl_weight_1_batch_size_128_epochs26.chkpt"
print(f"Loading model from: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval() # 设置为评估模式

# 字典 (与原脚本相同)
cdict = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))  
rev_dict = {j:i for i,j in cdict.items()}

N_SAMPLES_TO_GENERATE = 1000

# --- 3. 核心改动：从 QVAE 的玻尔兹曼机先验 p(z) 采样 ---

print(f"Sampling {N_SAMPLES_TO_GENERATE} sequences from QVAE prior (RBM)...")
with torch.no_grad():
    
    # 旧代码：sample = torch.randn(1000, 32) (已删除)

    # 新代码：
    # 1. 获取训练好的玻尔兹曼机 (先验)
    trained_prior = model.bm.to(device) 
    
    # 2. 使用采样器从先验中采样
    # 这将运行一个采样算法 (例如模拟退火) 来获取 p(z) 的样本
    # (这可能比 torch.randn 慢得多)
    z_samples = trained_prior.sample(sampler, n_samples=N_SAMPLES_TO_GENERATE)
    z_samples = z_samples.to(device) # 确保样本在 GPU 上以便解码
    
    # 3. 使用解码器解码离散样本
    sample_logits = model.decoder(z_samples)
    
    # 4. 将 logits 转换为概率 (Sigmoid) 并移回 CPU
    # (原 VAE 也使用了 sigmoid)
    sample_probs = torch.sigmoid(sample_logits).cpu()
    sample_probs = sample_probs.view(N_SAMPLES_TO_GENERATE, 70, 22)

# --- 4. 后处理 (与原脚本相同) ---

print("Decoding sequences...")
sampled_seqs = []
for i, seq_probs in enumerate(sample_probs):
    out_seq = []
    for j, pos_probs in enumerate(seq_probs):
        # 从概率中选择最可能的氨基酸
        best_idx = pos_probs.argmax()
        out_seq.append(rev_dict[best_idx.item()])
        
    final_seq = ''.join(out_seq).rstrip('0').rstrip('$')
    sampled_seqs.append(final_seq) 

seq_to_check = []
count = 0
for i in range(np.shape(sampled_seqs)[0]):
    # 过滤掉仍然包含 $ 或 0 的序列
    if sampled_seqs[i].find('$') == - 1 and sampled_seqs[i].find('0') == - 1:
        count = count + 1
        seq_to_check.append(['qvae_sample_'+str(count), sampled_seqs[i]])

print(f"Total valid sequences generated: {len(seq_to_check)}")
filtered_seq_to_check = pd.DataFrame(seq_to_check, columns = ['name', 'sequence'])
    
print('Total number of sequences:', len(filtered_seq_to_check))
filtered_seq_to_check = filtered_seq_to_check.drop_duplicates(subset='sequence').reset_index().drop('index', axis=1)
print('Total sequences remaining after duplicate removal', len(filtered_seq_to_check))

# 写入新的 FASTA 文件
output_fasta_name = 'qdata/amts_qvae_generated'
print(f"Writing final sequences to {output_fasta_name}.fasta")
write_fasta(output_fasta_name, filtered_seq_to_check)

print("Generation complete.")