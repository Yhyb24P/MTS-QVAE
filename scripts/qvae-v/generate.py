import pandas as pd
import numpy as np
import pickle # <-- 新增导入
from tqdm import tqdm
import re
from Bio import SeqIO
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.nn import functional as F
import torch
import argparse # 导入 argparse
import torch.multiprocessing as mp # <-- 新增导入

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

# --- 1. 定义 QVAE 组件 (类定义可以放在顶层) ---
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

# --- (核心修复) 必须使用 'if __name__ == "__main__":' ---
# 当使用 'spawn' 启动方法时，这是防止无限循环所必需的
if __name__ == "__main__":

    # --- (核心修复) 设置 'spawn' 启动方法 ---
    # 必须在任何 CUDA 或多进程代码之前调用
    try:
        # 'force=True' 确保即使它已被设置，我们也能覆盖它
        mp.set_start_method('spawn', force=True) 
        logging.info("已将 multiprocessing 启动方法设置为 'spawn' 以兼容 CUDA。")
    except RuntimeError as e:
        # 如果在特定环境 (如 Jupyter) 中运行，可能无法设置
        logging.warning(f"无法设置 'spawn' 启动方法 (可能已设置): {e}")
    # --- 结束修复 ---

    # --- 0. 添加命令行参数解析 ---
    parser = argparse.ArgumentParser(description="Generate sequences from a trained QVAE model.")
    parser.add_argument("--epoch", type=int, required=True, 
                        help="要加载的训练 epoch 编号。")
    parser.add_argument("--batch_size", type=int, default=2048, 
                        help="训练时使用的 BATCH_SIZE (用于定位 .chkpt 和 mean_x.pkl 文件)。")
    parser.add_argument("--n_samples", type=int, default=5000, 
                        help="要生成的序列数量。")
    args = parser.parse_args()

    # --- (新) 尽早获取参数 ---
    N_SAMPLES = args.n_samples
    EPOCH_TO_LOAD = args.epoch
    BATCH_SIZE_TRAINED = args.batch_size


    # --- 2. 实例化模型并加载权重 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # RBM (先验) 设置
    prior_vis = LATENT_DIM // 2
    prior_hid = LATENT_DIM - prior_vis

    # --- (动态加载 mean_x) ---
    # (路径更新) 匹配您新的 'model/qvae/' 目录
    mean_x_model_dir = "model/qvae" 
    mean_x_load_path = os.path.join(mean_x_model_dir, f"mean_x_bs{BATCH_SIZE_TRAINED}.pkl")
    
    try:
        with open(mean_x_load_path, 'rb') as f:
            mean_x_value = pickle.load(f)
        logging.info(f"成功从 {mean_x_load_path} 加载 train_bias (mean_x): {mean_x_value}")
    except FileNotFoundError:
        logging.warning(f"警告: 找不到 {mean_x_load_path}。")
        logging.warning("请确保您已运行 train_fixed.py (使用相同的 BATCH_SIZE) 来生成此文件。")
        logging.warning("将回退到硬编码值 (不推荐)...")
        mean_x_value = 0.04545454680919647 # (回退值)
    except Exception as e:
        logging.error(f"加载 mean_x 失败: {e}")
        sys.exit(1)
    # --- 结束 ---


    # 1. 实例化所有组件
    encoder = Encoder(INPUT_DIM, LATENT_DIM).to(device)
    decoder = Decoder(LATENT_DIM, INPUT_DIM).to(device)

    # 2.  实例化玻尔兹曼机先验 (RBM)
    bm_prior = RestrictedBoltzmannMachine(
        num_visible=prior_vis, # 16
        num_hidden=prior_hid   # 16
    ).to(device)
    logging.info(f"已初始化 RestrictedBoltzmannMachine 先验，总共 {bm_prior.num_nodes} 个节点。")

    # 3. (优化) 实例化强探索采样器
    logging.info("配置强探索采样器 (SimulatedAnnealingOptimizer)...")
    sampler = SimulatedAnnealingOptimizer(
        initial_temperature=500.0,
        alpha=0.999,
        cutoff_temperature=0.001,
        iterations_per_t=100,
        size_limit=N_SAMPLES,         # 在此处指定样本数量
        # --- (修复) 恢复 process_num=-1 ---
        # 既然我们使用了 'spawn'，我们现在可以安全地使用多进程
        process_num = -1
        # --- 结束修复 ---
    )
    logging.info(f"采样器进程数设置为: {sampler.process_num} (已启用 'spawn' 模式)")

    # 4. 实例化 QVAE 主模型
    model = QVAE(
        encoder=encoder,
        decoder=decoder,
        bm=bm_prior,
        sampler=sampler,
        dist_beta=1.0,           
        mean_x=mean_x_value, # 传递加载的或回退的 float 类型的 mean_x
        num_vis=bm_prior.num_visible 
    ).to(device)

    # 5. 加载训练好的权重
    # (路径更新) 匹配您新的 'model/qvae/' 目录
    model_load_path = f"model/qvae/qvae_mts_kl_weight_1_batch_size_{BATCH_SIZE_TRAINED}_epochs{EPOCH_TO_LOAD}.chkpt"
    logging.info(f"正在从 {model_load_path} 加载模型权重...")

    try:
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        logging.info("模型权重加载成功。")
    except FileNotFoundError:
        logging.error(f"错误: 找不到模型文件 {model_load_path}")
        logging.error("请确保 --epoch 和 --batch_size 参数与您训练保存的文件名匹配。")
        sys.exit(1) # 退出脚本

    model.eval() # 切换到评估模式

    # --- 3. 从先验 (RBM) 中采样隐变量 ---

    # 定义解码字典
    rev_dict = dict(zip(range(22), "FIWLVMYCATHGSQRKNEPD$0"))

    with torch.no_grad():
        logging.info(f"正在从 RBM 先验中采样 {N_SAMPLES} 个隐变量...")
        
        # --- (核心修复) 使用 RBM 模型的 .sample() 方法 ---
        # 根据您上传的 'abstract_boltzmann_machine.py' 源码，
        # RBM 基类有一个 'sample' 方法，该方法接收一个采样器 (sampler) 作为参数。
        # 它会:
        # 1. 调用 self.get_ising_matrix() (在内部转换为 CPU NumPy 数组)
        # 2. 调用 sampler.solve(ising_mat) (在 CPU 上使用多进程)
        # 3. 将 CPU 结果转换回 {0, 1} 并移回 RBM 所在的设备 (CUDA)
        #
        # 这正是我们需要的正确调用方式。
        try:
            # model.sampler (SimulatedAnnealingOptimizer) 在初始化时
            # 已经设置了 size_limit=N_SAMPLES
            z_samples = model.bm.sample(model.sampler)
            logging.info("采样完成。")
        except Exception as e:
            logging.error(f"调用 model.bm.sample(model.sampler) 时出错: {e}")
            logging.error("这可能表明 RBM、采样器或它们之间的交互存在问题。")
            logging.exception("详细错误信息:") # 打印完整的堆栈跟踪
            sys.exit(1)
        # --- 结束修复 ---
        
        z_samples = z_samples.to(device) # 确保样本在正确的设备上 (虽然 .sample() 应该已经做了)
        
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
    # (路径更新) 匹配您的模型路径 'model/qvae/'
    output_dir = "data/qvae-v/output" 
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"generated_seqs_e{EPOCH_TO_LOAD}_b{BATCH_SIZE_TRAINED}_n{N_SAMPLES}.fasta")

    with open(output_filename, 'w') as f:
        for index, row in filtered_seq_to_check.iterrows():
            f.write(">" + row['name'] + "\n")
            f.write(row['sequence'] + "\n")

    logging.info(f"已将 {len(filtered_seq_to_check)} 条生成的序列保存到: {output_filename}")

