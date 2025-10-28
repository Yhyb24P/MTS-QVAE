# -*- coding: utf-8 -*-
"""
此脚本用于分析 `model_organism_sequences_mts.fasta` 文件中
四个目标物种 (YEAST, Rhoto, HUMAN, TOBAC) 的序列分布情况。

它将执行《QVAE 架构优化指南》中的“策略一：分析数据”。

它会计算两个指标：
1.  Header Counts: 基于 FASTA 标题中关键字的物种总数。
2.  Valid 'M' Counts: 标题匹配且序列以 'M' (甲硫氨酸) 开头的物种总数。
    (注意：`sample.py` 脚本在分析时会过滤掉不以 'M' 开头的序列，
     因此 'Valid M Counts' 是最能反映模型训练不平衡的指标。)
"""

try:
    # 导入 Biopython 库，`sample.py` 中也使用了它
    from Bio import SeqIO
except ImportError:
    print("错误: 未找到 Biopython 库。")
    print("请先安装: pip install biopython")
    exit()

import sys

# --- 配置 ---
FASTA_FILE_PATH = 'data/model_organism_sequences_mts.fasta'
SPECIES_KEYWORDS = ['YEAST', 'Rhoto', 'HUMAN', 'TOBAC']
# ----------------

def analyze_distribution(fasta_file):
    """
    解析 FASTA 文件并统计物种分布。
    """
    # 初始化计数器
    header_counts = {key: 0 for key in SPECIES_KEYWORDS + ['OTHER']}
    valid_m_counts = {key: 0 for key in SPECIES_KEYWORDS + ['OTHER']}
    total_records = 0
    total_m_records = 0

    try:
        # 使用 Biopython 的 SeqIO.parse 来安全地读取 FASTA 文件
        for record in SeqIO.parse(fasta_file, 'fasta'):
            total_records += 1
            header = record.id  # 序列的 ID/Header
            sequence = str(record.seq).strip() # 序列字符串

            # 1. 检查 Header 属于哪个物种
            found_species = 'OTHER'
            for key in SPECIES_KEYWORDS:
                if key in header:
                    found_species = key
                    break
            
            header_counts[found_species] += 1

            # 2. 检查序列是否有效 (以 'M' 开头)
            if sequence and sequence.startswith('M'):
                valid_m_counts[found_species] += 1
                total_m_records += 1

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{fasta_file}'")
        print("请确保此脚本与 .fasta 文件位于同一目录中。")
        sys.exit(1)
    except Exception as e:
        print(f"解析 FASTA 文件时出错: {e}")
        sys.exit(1)

    return header_counts, valid_m_counts, total_records, total_m_records

def print_report(title, counts, total):
    """
    打印格式化的分布报告。
    """
    print(f"\n--- {title} (总计: {total}) ---")
    if total == 0:
        print("未找到相关序列。")
        return

    for species, count in counts.items():
        percentage = (count / total) * 100
        print(f"  {species:<10}: {count:>6} 条 ({percentage:6.2f}%)")

# --- 主程序 ---
if __name__ == "__main__":
    print(f"正在分析 '{FASTA_FILE_PATH}'...")
    
    h_counts, m_counts, total_seqs, total_m_seqs = analyze_distribution(FASTA_FILE_PATH)
    
    # 报告1: 基于 Header 的总分布
    print_report("物种总分布 (基于 Header 关键字)", h_counts, total_seqs)
    
    # 报告2: 实际用于分析的分布 (以 'M' 开头的序列)
    # 这是最重要的报告！
    print_report("有效序列分布 (以 'M' 开头)", m_counts, total_m_seqs)
    
    print("\n--- 分析完毕 ---")
    if total_m_seqs > 0:
        print("请重点关注“有效序列分布 (以 'M' 开头)”报告。")
        print("如果 HUMAN 和 TOBAC 的百分比远低于 Rhoto 和 YEAST，")
        print("则“数据不平衡”假说成立。")
