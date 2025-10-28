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

# --- VAE 模型定义 ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.relu = nn.ReLU()
        # 编码器层
        self.fc1 = nn.Linear(1540, 512)
        self.fc21 = nn.Linear(512, 32)
        self.fc22 = nn.Linear(512, 32)
        # 解码器层
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
        """
        前向传播。
        注意：此 `forward` 方法在 `generate.py` 中实际并未使用。
        生成过程是直接调用 `model.decode()`。
        """
        mu, logvar = self.encode(x.view(-1, 1540))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

model = VAE() # 实例化 VAE 模型

# --- 加载预训练模型 ---
# 加载在 train.py 中训练好的模型权重 (epoch 32 被选为最佳模型)
model_path = "model/vae/vae_self_tv_sim_split_kl_weight_1_batch_size_128_epochs32.chkpt"
model.load_state_dict(torch.load(model_path))
model.eval() # 设置模型为评估模式

# --- 字典定义 ---
# 字符到索引的映射
cdict = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))  
# 索引到字符的反向映射 (用于解码)
rev_dict = {j:i for i,j in cdict.items()}

# --- 序列生成 (采样) ---
print("--- 开始从潜在空间采样 ---")
with torch.no_grad(): # 禁用梯度计算
    sample = torch.randn(1000, 32)
    sample = model.decode(sample).cpu() 
    sample = sample.view(1000, 70, 22)

sampled_seqs = []
for i, seq in enumerate(sample): # 遍历 1000 个生成的样本
    out_seq = []
    for j, pos in enumerate(seq): # 遍历 70 个位置
        
        best_idx = pos.argmax() 
        out_seq.append(rev_dict[best_idx.item()])
        
    final_seq = ''.join(out_seq).rstrip('0').rstrip('$')
    sampled_seqs.append(final_seq) 

# --- 序列过滤 ---
print("--- 过滤生成的序列 ---")
seq_to_check = []
count = 0
for i in range(np.shape(sampled_seqs)[0]):
    if sampled_seqs[i].find('$') == - 1 and sampled_seqs[i].find('0') == - 1:
        count = count + 1
        seq_to_check.append(['sample'+str(count), sampled_seqs[i]]) 

print(f"生成的有效序列数 (过滤前): {np.shape(seq_to_check)[0]}")
# 将有效序列列表转换为 pandas DataFrame
filtered_seq_to_check = pd.DataFrame(seq_to_check, columns = ['name', 'sequence'])
    
print('总序列数:', len(filtered_seq_to_check))
# 8. 去除重复的序列
filtered_seq_to_check = filtered_seq_to_check.drop_duplicates(subset='sequence').reset_index().drop('index', axis=1)
print('去重后剩余的总序列数:', len(filtered_seq_to_check))

# 9. 将最终的、唯一的、有效的序列写入 FASTA 文件
output_fasta_path = 'data/vae/output/amts'
write_fasta(output_fasta_path, filtered_seq_to_check)
print(f"--- 生成的序列已保存到 {output_fasta_path}.fasta ---")
