import numpy as np
import pandas as pd
import umap  # 导入 UMAP 库，用于非线性降维
import pickle
from Bio import SeqIO  # 导入 BioPython 库，用于读写 FASTA 文件
import re  # 导入正则表达式库

def validate(seq, pattern=re.compile(r'^[FIWLVMYCATHGSQRKNEPD]+$')):
    """
    验证序列是否仅包含标准的20种氨基酸。
    """
    if (pattern.match(seq)):
        return True
    return False

def clean(sequence_df):
    """
    (此函数在脚本中未被实际调用)
    清理 DataFrame 中的无效序列。
    注意：代码中引用了 'zf_sequence' 列，这可能是一个拼写错误或遗留代码。
    """
    invalid_seqs = []

    for i in range(len(sequence_df)):
        if (not validate(sequence_df['zf_sequence'][i])):
            invalid_seqs.append(i)

    print('Total number of sequences dropped:', len(invalid_seqs))
    sequence_df = sequence_df.drop(invalid_seqs).reset_index().drop('index', axis=1)
    print('Total number of sequences remaining:', len(sequence_df))
    
    return sequence_df

def read_fasta(name):
    """
    读取 FASTA 文件并将其内容转换为 [id, sequence] 列表。
    """
    fasta_seqs = SeqIO.parse(open(name + '.fasta'),'fasta')
    data = []
    for fasta in fasta_seqs:
        data.append([fasta.id, str(fasta.seq).strip()]) # 提取 ID 和序列
            
    return data

# --- 1. 加载和处理 "标准" 生物体的 MTS 序列 (Ground Truth) ---

in_fasta = 'data/model_organism_sequences_mts'
seqs_df_total = pd.DataFrame(read_fasta(in_fasta), columns = ['name', 'sequence'])

# 根据序列名称中的关键词为序列分配物种标签
org_label = []  # 物种标签 (1-4)
index_label = [] # 保留的序列索引
for i in range(np.shape(seqs_df_total)[0]):
    if 'YEAST' in seqs_df_total['name'][i]: # 酵母
        org_label.append(1)
        index_label.append(i)
    elif 'Rhoto' in seqs_df_total['name'][i]: # 另一个酵母/真菌
        org_label.append(2)
        index_label.append(i)
    elif 'HUMAN' in seqs_df_total['name'][i]: # 人类
        org_label.append(3)
        index_label.append(i)
    elif 'TOBAC' in seqs_df_total['name'][i]: # 烟草 (植物)
        org_label.append(4)
        index_label.append(i)
        
# 仅保留被标记的序列
seqs_df = seqs_df_total.iloc[index_label].reset_index(drop=True)
seqs_df['label'] = org_label # 添加标签列

# 过滤：只保留以 'M' (甲硫氨酸，起始密码子) 开头的序列
final_seqs_df = seqs_df[seqs_df.sequence.str.startswith('M')]

import matplotlib.pyplot as plt

# 可视化：绘制各类物种序列的数量
fig, ax = plt.subplots()
final_seqs_df['label'].value_counts().plot(ax=ax, kind='bar')

# --- 2. 加载 "标准" 序列的预计算嵌入 (Embeddings) ---

# 加载 .npz 文件，其中包含序列的预计算嵌入向量 (可能来自 UniRep 或其他模型)
arrays = np.load('data/model_organism_sequences_mts.npz', allow_pickle=True) 

embd_for_cluster = [] # 用于聚类的嵌入向量
cluster_data_embd_arranged = [] # 重新排序后的序列信息
for i in list(arrays.keys()):
    if i in list(final_seqs_df['name']): # 确保嵌入向量对应的序列在过滤后的 DataFrame 中
        cluster_data_embd_arranged.append(list(final_seqs_df.loc[final_seqs_df['name'] == i].values[0]))
        embd_for_cluster.append(arrays[i].item()['avg']) # 提取平均嵌入向量
        
cluster_data_embd_arranged_df = pd.DataFrame(cluster_data_embd_arranged, columns = ['name','sequence','label'])

# --- 3. UMAP 降维和聚类中心计算 ---

reducer = umap.UMAP(n_components=2) # 初始化 UMAP，降到 2 维
embedding = reducer.fit_transform(embd_for_cluster) # 对标准序列的嵌入进行降维

# 将 2D 嵌入结果存入 DataFrame
df_cluster_rd_unirep_umap = pd.DataFrame(data=embedding, columns=['x1','x2'])

# 计算每个物种的 "聚类中心" (嵌入向量的均值)
cluster_center = []
org_number = 4 # 物种数量
for i in range(org_number):
    curr_index = cluster_data_embd_arranged_df.index[cluster_data_embd_arranged_df['label'] == i+1].tolist()
    for j in range(len(curr_index)):
        if j == 0:
            cc_vector = embd_for_cluster[curr_index[j]]
        else:
            cc_vector = cc_vector + embd_for_cluster[curr_index[j]]
    
    # 计算均值向量
    if i == 0:
        cluster_center = cc_vector/len(curr_index)
    else:
        cluster_center = np.vstack((cluster_center,cc_vector/len(curr_index)))
        
print(cluster_center.shape) # 应为 (4, embedding_dim)

# --- 4. 加载 "人工" 生成的 MTS 序列 (来自 generate.py) ---


in_fasta = 'data/amts' 
ini_amts_df = pd.DataFrame(read_fasta(in_fasta), columns = ['name', 'sequence'])

# 同样过滤，只保留 'M' 开头的序列
amts_df = ini_amts_df[ini_amts_df.sequence.str.startswith('M')]

# 加载人工序列的预计算嵌入
arrays = np.load('data/amts' + '.npz', allow_pickle=True) 
embd_for_amts = []
amts_data_embd_arranged = []
for i in list(arrays.keys()):
    if i in list(amts_df['name']):
        amts_data_embd_arranged.append(list(amts_df.loc[amts_df['name'] == i].values[0]))
        embd_for_amts.append(arrays[i].item()['avg'])
        
amts_data_embd_arranged_df = pd.DataFrame(amts_data_embd_arranged, columns = ['name','sequence'])

# --- 5. "采样算法" (论文图 3a 所示) ---
# 目标：为人工序列(amts)分配物种标签
from scipy.spatial import distance # 导入距离计算库

amts_label = [] # 存储分配的标签
selected_amts_index = [] # 存储通过筛选的序列索引

for i in range(np.shape(amts_df)[0]): # 遍历每条人工序列
    
    # --- Criterion 1: 欧氏距离 (到聚类中心) ---
    curr_d = []
    for j in range(org_number):
        # 计算当前人工序列到 4 个物种聚类中心的距离
        curr_d.append(distance.euclidean(embd_for_amts[i], cluster_center[j])) 
    
    # --- Criterion 2: k-Nearest Neighbors (k=20) ---
    curr_d_density = []
    for j in range(np.shape(embd_for_cluster)[0]):
        # 计算当前人工序列到 *所有* 标准序列的距离
        curr_d_density.append(distance.euclidean(embd_for_amts[i], embd_for_cluster[j])) 
        
    # 找到最近的 20 个邻居
    top_name_list = list(cluster_data_embd_arranged_df['name'][np.argpartition(curr_d_density, 20)[:20]])
    
    # 统计 20 个邻居中每个物种的数量
    c1 = sum('YEAST' in s for s in top_name_list)
    c2 = sum('Rhoto' in s for s in top_name_list)
    c3 = sum('HUMAN' in s for s in top_name_list)
    c4 = sum('TOBAC' in s for s in top_name_list)
    count_organism = [c1, c2, c3, c4]
    
    # --- 筛选 ---
    # 必须同时满足两个条件：
    # 1. 离聚类中心最近的物种 (np.argmin(curr_d))
    # 2. k-NN 中占比最高的物种 (np.argmax(count_organism))
    # ... 必须是同一个物种
    if np.argmin(curr_d) == np.argmax(count_organism):
        selected_amts_index.append(i) # 保存索引
        amts_label.append(np.argmin(curr_d)+1) # 分配标签 (1-4)
        
# 根据筛选结果创建新的 DataFrame
amts_data_embd_arranged_cf = amts_data_embd_arranged_df.iloc[selected_amts_index].reset_index(drop=True)
amts_data_embd_arranged_cf['org_label'] = amts_label

embd_for_amts_clustering = list(np.array(embd_for_amts)[selected_amts_index])

# 可视化：绘制被选中的人工序列的物种标签分布
fig, ax = plt.subplots()
amts_data_embd_arranged_cf['org_label'].value_counts().plot(ax=ax, kind='bar')

# (这部分代码块重复了，功能同上)
# embd_for_amts_clustering = list(np.array(embd_for_amts)[selected_amts_index])
# fig, ax = plt.subplots()
# amts_data_embd_arranged_cf['org_label'].value_counts().plot(ax=ax, kind='bar')

# --- 6. 可视化 UMAP 聚类结果 (论文图 3a) ---

# 使用训练好的 reducer 将 4 个聚类中心也转换到 2D 空间
cc_unirep_umap = reducer.transform(cluster_center)
cc_unirep_umap = pd.DataFrame(data=cc_unirep_umap, columns=['x1', 'x2'])
print(cc_unirep_umap.shape)

from scipy.stats import gaussian_kde # 用于计算核密度估计
import matplotlib.cm as cm 

fig, ax = plt.subplots()
# 遍历 4 个物种标签
for g in np.unique(cluster_data_embd_arranged_df['label']):
    ix = np.where(cluster_data_embd_arranged_df['label'] == g)
    x = df_cluster_rd_unirep_umap['x1'].loc[ix]
    y = df_cluster_rd_unirep_umap['x2'].loc[ix]
    
    # 计算数据点的密度 (用于颜色深浅)
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # 根据物种使用不同的颜色映射 (cmap)
    if g == 1: # YEAST
        ax.scatter(x, y, c = z, cmap=cm.RdPu, s = 10, alpha = 0.4)
    elif g == 2: # Rhoto
        ax.scatter(x, y, c = z, cmap=cm.Oranges, s = 10, alpha = 0.4)
    elif g == 3: # HUMAN
        ax.scatter(x, y, c = z, cmap=cm.Blues, s = 10, alpha = 0.4)
    else: # TOBAC
        ax.scatter(x, y, c = z, cmap=cm.Greens, s = 10, alpha = 0.4)
        
# 在图上用星号 (*) 标出 4 个物种的聚类中心
cdict = {1: 'magenta', 2: 'orange', 3: 'blue', 4: 'green'}
for g in np.unique(cluster_data_embd_arranged_df['label']):
    ax.scatter(cc_unirep_umap['x1'][g-1], cc_unirep_umap['x2'][g-1], c = cdict[g], label = g, marker = '*', s = 200)

plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('data/clustering_amts_to_test', dpi=400) # 保存图像

# --- 7. 保存最终筛选和标记的人工序列 ---

# 将数字标签 (1-4) 转换回物种名称
org_label = []
for i in range(np.shape(amts_data_embd_arranged_cf)[0]):
    if amts_data_embd_arranged_cf['org_label'][i] == 1:
        org_label.append('YEAST')
    elif amts_data_embd_arranged_cf['org_label'][i] == 2:
        org_label.append('Rhoto')
    elif amts_data_embd_arranged_cf['org_label'][i] == 3:
        org_label.append('HUMAN')
    elif amts_data_embd_arranged_cf['org_label'][i] == 4:
        org_label.append('TOBAC')
        
amts_data_embd_arranged_cf['org_label'] = org_label

# 将最终结果保存为 CSV 文件
amts_data_embd_arranged_cf.to_csv('data/amts_labeled_cluster_final.csv',index=False)