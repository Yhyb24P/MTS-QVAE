import pandas as pd
import numpy as np
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import logging
import os

# --- 设置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- GFP 序列 ---
# 这是来自 DeepLoc 1.0 论文中使用的 GFP 报告蛋白序列
# (移除了起始的甲硫氨酸 'M'，因为它将被我们的 MTS 替换)
GFP_SEQUENCE = (
    "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYG"
    "VQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKE"
    "DGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPD"
    "NHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
)

def main():
    parser = argparse.ArgumentParser(
        description="将生成的MTS序列与GFP报告蛋白融合，用于DeepLoc提交。"
    )
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="输入的FASTA文件路径 (包含您生成的MTS序列)。"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="输出的FASTA文件路径 (用于提交到DeepLoc)。"
    )
    args = parser.parse_args()

    logging.info(f"正在从 {args.input} 读取生成的序列...")
    
    try:
        generated_records = list(SeqIO.parse(args.input, "fasta"))
    except FileNotFoundError:
        logging.error(f"错误: 找不到输入文件 {args.input}")
        return
    except Exception as e:
        logging.error(f"读取FASTA文件时出错: {e}")
        return

    if not generated_records:
        logging.warning(f"输入文件 {args.input} 中没有找到序列。")
        return

    logging.info(f"找到了 {len(generated_records)} 条序列。")
    logging.info(f"将使用 {len(GFP_SEQUENCE)} AA 的GFP序列进行融合。")

    fused_records = []
    for record in generated_records:
        mts_seq = str(record.seq)
        
        # 融合: MTS 序列 + GFP 序列
        fused_seq_str = mts_seq + GFP_SEQUENCE
        
        # 创建新的 SeqRecord
        fused_record = SeqRecord(
            Seq(fused_seq_str),
            id=f"{record.id}_GFP_fused",
            description=f"Fused protein: {record.id} + GFP reporter"
        )
        fused_records.append(fused_record)

    try:
        SeqIO.write(fused_records, args.output, "fasta")
        logging.info(f"成功！已将 {len(fused_records)} 条融合序列保存到: {args.output}")
        logging.info("您现在可以提交此文件到 DeepLoc 2.0 (或 1.0) 进行亚细胞定位预测。")
    except Exception as e:
        logging.error(f"写入输出文件时出错: {e}")

if __name__ == "__main__":
    main()

