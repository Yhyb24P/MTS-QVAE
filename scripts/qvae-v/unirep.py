import numpy as np
from jax_unirep import get_reps
from tqdm import tqdm
import sys

# --- 1. 定义一个简单的 FASTA 读取器  ---
def simple_fasta_reader(fasta_file):
    """从 FASTA 文件读取序列并返回一个字典 {name: sequence}"""
    sequences = {}
    current_name = None
    try:
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    current_name = line[1:].split()[0] 
                    sequences[current_name] = ""
                elif current_name:
                    sequences[current_name] += line
        return sequences
    except FileNotFoundError:
        print(f"错误: 输入文件未找到: {fasta_file}", file=sys.stderr)
        sys.exit(1)

# --- 2. 定义输入和输出文件路径 ---
INPUT_FASTA = 'data/qvae-v/output/generated_seqs_best_b2048_n5000.fasta' 

OUTPUT_NPZ = 'data/qvae-v/output/generated_seqs_best_b2048_n5000.npz' 

# --- 3. 加载序列 ---
print(f"正在从 {INPUT_FASTA} 加载序列...")
sequences_dict = simple_fasta_reader(INPUT_FASTA)
names_list = list(sequences_dict.keys())
seqs_list = list(sequences_dict.values())

if not seqs_list:
    print(f"在 {INPUT_FASTA} 中未找到序列。程序退出。", file=sys.stderr)
    sys.exit(1)

print(f"已找到 {len(seqs_list)} 条序列。")

# --- 4. 计算 UniRep 嵌入 ---
print("正在计算 UniRep 嵌入 (这可能需要几分钟)...")

# get_reps 返回一个元组 (avg_reps, h_reps)
avg_embeddings = get_reps(seqs_list)[0] 
print(f"嵌入计算完成。 Shape: {avg_embeddings.shape}")

# --- 5. 格式化为 sample.py 所需的字典 ---
output_data = {}
for i in tqdm(range(len(names_list)), desc="格式化 .npz 文件"):
    name = names_list[i]
    vector = avg_embeddings[i]
    # 注意: .item()['avg'] 意味着值必须是一个字典
    output_data[name] = {'avg': vector} 

# --- 6. 保存为 .npz 文件 ---
print(f"正在将嵌入保存到 {OUTPUT_NPZ}...")

# 使用 np.savez 将字典保存为 .npz 文件
np.savez(OUTPUT_NPZ, **output_data) 

print("转换完成")
