import pandas as pd
import os
import sys

def write_fasta(name, sequence_df):
    """
    将包含 'name' 和 'sequence' 列的 DataFrame 写入 FASTA 文件。
    """
    try:
        with open(name + '.fasta', "w") as out_file:
            for i in range(len(sequence_df)):
                # 使用 .iloc 确保按行索引访问
                out_file.write('>' + sequence_df.iloc[i]['name'] + '\n')
                out_file.write(sequence_df.iloc[i]['sequence'] + '\n')
        print(f"成功将 {len(sequence_df)} 条序列写入 {name}.fasta")
    except Exception as e:
        print(f"写入 FASTA 文件时出错: {e}")
        sys.exit()

# --- 1. 定义筛选阈值 ---
MIN_MTP_PROBABILITY = 0.8  # 筛选 mTP 概率 > 80% 的序列

# --- 2. 定义文件路径 ---
TARGETP_FILE = 'qdata/output_protein_type.txt'

# 这个 CSV 文件是由 sample.py 生成的，包含了完整的 MTS-GFP 序列
SEQUENCES_FILE = 'qdata/amts_labeled_cluster_final.csv'
OUTPUT_FILE_NAME = f'qdata/final_successful_candidates_mTP_gt_{int(MIN_MTP_PROBABILITY*100)}'

print(f"开始筛选。目标：找到 TargetP 预测为 'mTP' 且概率 > {MIN_MTP_PROBABILITY} 的序列。")

# --- 3. 加载 TargetP 预测结果 ---
if not os.path.exists(TARGETP_FILE):
    print(f"错误: 找不到 TargetP 输出文件: {TARGETP_FILE}")
    print("请确保该文件与此脚本位于同一目录中。")
    sys.exit()

try:
    # === ★★★ 关键修改 ★★★ ===
    # 1. 移除 comment='#'，因为它会跳过我们需要的表头。
    # 2. 添加 header=1，明确告诉 Pandas 文件的第二行 (索引 1) 才是表头。
    targetp_df = pd.read_csv(TARGETP_FILE, sep='\t', header=1)
    
    # 3. 重命名第一列 (从 '# ID' 改为 'ID') 以便后续合并
    targetp_df.rename(columns={'# ID': 'ID'}, inplace=True)
    # === 修改结束 ===
    
    print(f"成功加载 TargetP 结果: {TARGETP_FILE} (共 {len(targetp_df)} 条记录)")
except Exception as e:
    print(f"读取 TargetP 文件时出错: {e}")
    sys.exit()

# --- 4. 加载完整的序列数据 ---
if not os.path.exists(SEQUENCES_FILE):
    print(f"错误: 找不到序列文件: {SEQUENCES_FILE}")
    print("请确保您已经先运行了 'sample.py' 来生成此文件。")
    sys.exit()
    
try:
    sequences_df = pd.read_csv(SEQUENCES_FILE)
    print(f"成功加载序列文件: {SEQUENCES_FILE} (共 {len(sequences_df)} 条序列)")
except Exception as e:
    print(f"读取序列文件时出错: {e}")
    sys.exit()

# --- 5. 执行筛选 ---
# (此部分现在应该可以正常工作了)
if 'Prediction' not in targetp_df.columns or 'mTP' not in targetp_df.columns:
    print(f"错误: TargetP 文件 {TARGETP_FILE} 缺少 'Prediction' 或 'mTP' 列。")
    sys.exit()

# 步骤 5a: 仅保留 TargetP 预测为 'mTP' 的序列
successful_candidates = targetp_df[targetp_df['Prediction'] == 'mTP'].copy()
print(f"找到 {len(successful_candidates)} 条被预测为 'mTP' 的序列。")

if len(successful_candidates) == 0:
    print("未找到任何被预测为 'mTP' 的序列。脚本将退出。")
    sys.exit()

# 步骤 5b: 筛选出概率高于阈值的序列
strong_candidates = successful_candidates[successful_candidates['mTP'] > MIN_MTP_PROBABILITY]
print(f"其中 {len(strong_candidates)} 条序列的 mTP 概率 > {MIN_MTP_PROBABILITY}。")

if len(strong_candidates) == 0:
    print("未找到符合条件的高置信度序列。脚本将退出。")
    sys.exit()
    
# --- 6. 合BING数据以获取完整序列 ---
# (此部分现在也应该可以正常工作了)
if 'ID' not in strong_candidates.columns or 'name' not in sequences_df.columns:
    print("错误: TargetP 文件缺少 'ID' 列或序列文件缺少 'name' 列。")
    sys.exit()


# 使用 'ID' 和 'name' 作为键进行合并
final_df = pd.merge(strong_candidates, sequences_df, left_on='ID', right_on='name')

if len(final_df) == 0:
    print("合并失败。TargetP ID 和序列名称之间没有匹配项。")
    sys.exit()

# --- 7. 排序并保存到 FASTA ---
# 按 mTP 概率降序排列
final_df = final_df.sort_values(by='mTP', ascending=False)

# 确保 'name' 和 'sequence' 列存在
if 'name' not in final_df.columns or 'sequence' not in final_df.columns:
    print("错误：合并后的 DataFrame 中缺少 'name' 或 'sequence' 列。")
    sys.exit()

# 确保 qdata 目录存在
os.makedirs('qdata', exist_ok=True)

# 写入 FASTA
write_fasta(OUTPUT_FILE_NAME, final_df[['name', 'sequence']])

# 额外保存一个 CSV 以供审查
# 仅选择相关列
columns_to_save = ['ID', 'Prediction', 'mTP', 'OTHER', 'SP', 'sequence']
final_df[columns_to_save].to_csv(OUTPUT_FILE_NAME + '.csv', index=False)
print(f"已将详细信息保存到 {OUTPUT_FILE_NAME}.csv")
print("筛选完成。")

