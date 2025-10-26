(.venv)环境配置
创建并激活`unirep` 环境
```
python3 -m venv unirep
source unirep/bin/activate
```
安装依赖
```
# 升级 pip
python -m pip install -U pip
# 安装 unirep 依赖
pip install -r requirements_unirep.txt -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
```
创建并激活`qvae`环境
```
# 注意：确保使用 Python 3.10
python3.10 -m venv qvae
source qvae/bin/activate
```
安装依赖
```
# 升级 pip
python -m pip install -U pip
# 安装 qvae 依赖
# (requirements.txt 包含 PyTorch+cu130, kaiwu, umap 等)
pip install -r requirements_qvae.txt -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple) --extra-index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)
```
## 项目文件架构
关键文件组织如下：
```
.
├── kaiwu-1.2.0-cp310-none-manylinux1_x86_64.whl  (Kaiwu 核心库)
├── kaiwu_torch_plugin/                           (Kaiwu Pytorch 插件)
├── model/                                        (存放训练好的模型权重)
├── qdata/                                        (存放 one-hot 编码数据和生成序列)
├── scripts/
│   ├── qvae/                                     (QVAE 核心脚本)
│   │   ├── train.py                          (QVAE 训练脚本)
│   │   └── generate.py                       (QVAE 序列生成脚本)
│   ├── sample.py                             (可视化验证脚本)
│   └── unirep.py                             (unirep 数据转换脚本)
├── unirep/                                       (unirep 库)
├── README.md                                     (本项目说明)
├── requirements_qvae.txt                         (qvae 环境依赖)
├── requirements_unirep.txt                       (unirep 环境依赖)
└── ... (其他配置文件)
```
## 模型架构演进：从 VAE 到 QBM-VAE
本项目的核心模型经历了一次重要演进。最初的实现是一个标准的变分自编码器 (VAE)，而当前版本 (`scripts/qvae/train.py`) 则是一个更强大的**量子启发玻尔兹曼机 VAE (QBM-VAE 或 QVAE)**。
### 核心动机：超越高斯先验
标准 VAE 假设潜变量 $z$ 遵从一个简单的、固定的**高斯先验**（即 $\mathcal{N}(0, I)$）。这是一个“单峰”分布，难以捕捉真实世界数据（如蛋白质序列空间）中复杂的、**多模态 (multi-modal)** 的内在结构。
为了克服这一局限性，我们将模型升级为 QVAE，其核心思想是：
**用一个更强大的、可学习的玻尔兹曼机 (BM) 分布来替换简单的高斯先验。**
### 关键技术转变
这次演进涉及了模型架构、训练逻辑和所用库的根本性变化：

| **特性**             | **原始标准 VAE (旧脚本)**            | **当前 QVAE (train.py)**                      |
| ------------------ | ----------------------------- | ------------------------------------------- |
| **核心库**            | 纯 PyTorch (`torch.nn.Module`) | `kaiwu_torch_plugin`                        |
| **潜变量先验** $p(z)$   | 固定的高斯分布                       | **可学习的** `RestrictedBoltzmannMachine` (RBM) |
| **潜变量** $z$ **类型** | 连续 (Continuous)               | 二元 (Binary / Discrete)                      |
| **编码器输出**          | `mu` (均值) 和 `logvar` (方差)     | `q_logits` (二元概率的 logits)                   |
| **损失函数**           | 重建损失 (BCE) + KL 散度 (vs 高斯)    | `model.neg_elbo`                            |
| **训练逻辑**           | 单一梯度流                         | **双重梯度流** (VAE参数 + RBM先验参数)                 |
| **采样器**            | 高斯再参数化                        | **`SimulatedAnnealingOptimizer`** (模拟退火)    |

**深度解读：**
1. **RBM 先验**：我们现在使用 `RestrictedBoltzmannMachine` 来定义先验 $p(z)$。这是一个基于能量的模型，理论上可以学习并表示复杂的多模态分布，更适合描述蛋白质序列的潜空间。
    
2. **双重梯度流**：训练不再是简单的 `loss.backward()`。`model.neg_elbo` 内部封装了复杂的**对比散度 (CD)** 算法。它同时优化两组参数：
    
    - VAE 的编码器/解码器参数。
        
    - RBM 先验自身的能量参数（权重和偏置）。
        
3. **经典采样器**：理论上，从 RBM 中采样（获取“负相”样本）计算上是困难的。虽然模型架构受量子玻尔兹曼机启发，但我们在 `train.py` 中使用了一个强大的**经典**算法——`SimulatedAnnealingOptimizer`（模拟退火）——来高效地从 RBM 先验中采样。
    
### 改进`generate.py`：从先验生成新序列

为了利用这个更强大的 RBM 先验，我们改进了 `scripts/qvae/generate.py` 脚本。
- **功能**：此脚本不再像标准 VAE 那样从 $\mathcal{N}(0, I)$ 中采样，而是直接从**已训练好的 RBM 先验模型 (`model.bm`)** 中采样。
    
- **流程**：
    
    1. 加载 `train.py` 训练好的 QVAE 权重。
        
    2. 提取 RBM 先验 (`model.bm`)。
        
    3. 使用 `SimulatedAnnealingOptimizer` 从 RBM 先验中采样，生成一批**二元**潜变量 $z$。
        
    4. 将这些 $z$ 样本送入 QVAE 的**解码器 (`model.decoder`)**，生成新的 one-hot 编码序列。
        
    5. 解码为 FASTA 格式并保存。
        
## 工作流程
基于上述演进，项目的完整工作流程如下：
```
# 1. 激活 qvae 环境
source qvae/bin/activate

# 2. 训练 QVAE 模型 (使用 RBM 先验和 SA 采样器)
python scripts/qvae/train.py

# 3. [新增] 从训练好的 RBM 先验中生成新序列
python scripts/qvae/generate.py

# 4. 退出当前环境，打开 unirep 环境
deactivate
source unirep/bin/activate

# 5. [可选] 运行 unirep 进行数据转换 (如需)
python scripts/unirep.py

# 6. 退出当前环境，打开 qvae 环境
deactivate
source qvae/bin/activate

# 7. [可选] 进行降维与可视化验证
python scripts/sample.py
```
