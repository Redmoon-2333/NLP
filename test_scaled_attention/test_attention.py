"""
缩放点积注意力测试模块

功能描述:
    演示Transformer中缩放因子 √d_k 的作用。
    通过对比缩放前后的梯度变化，验证缩放操作的必要性。

核心原理:
    当 d_k 较大时，Q·K^T 的点积值会很大，导致 softmax 进入饱和区。
    饱和区的梯度接近于0，导致训练困难。
    缩放操作：scores = Q·K^T / √d_k，将点积值控制在合理范围。

作者: Red_Moon
创建日期: 2026-02
"""

import torch

# 禁用科学计数法显示（意图：便于观察数值变化）
torch.set_printoptions(sci_mode=False)

# ==================== 参数配置 ====================
seq_len = 10    # 序列长度
dk = 64         # Key/Query的维度（Transformer原论文中 d_k = d_model / num_heads = 512/8 = 64）

# ==================== 第一组实验：不使用缩放 ====================
# 意图：展示不缩放时梯度的问题

# 创建Query和Key矩阵，需要梯度以便观察反向传播
# Q.shape: [seq_len, dk] = [10, 64]
# K.shape: [seq_len, dk] = [10, 64]
Q = torch.randn([seq_len, dk], requires_grad=True)
K = torch.randn([seq_len, dk], requires_grad=True)

# 计算注意力分数（未缩放）
# scores = Q @ K^T
# scores.shape: [seq_len, seq_len] = [10, 10]
# 【警示】当dk=64时，点积的方差约为dk=64，标准差约为8
#         这意味着点积值可能很大（如±20甚至更大）
scores = Q @ K.T

# Softmax归一化
# weights.shape: [seq_len, seq_len]
# 【关键问题】当分数值很大时，softmax输出接近one-hot向量
#             例如：[20, 5, 3, 1] -> softmax -> [0.9999, 0.0001, ~0, ~0]
#             这种情况下梯度几乎为0，模型难以学习
weights = torch.softmax(scores, dim=-1)
print(weights)

# 反向传播测试
# 取weights[0,0]作为loss，计算梯度
loss = weights[0,0]
loss.backward()

# 打印Q的梯度范数
# 【观察点】梯度范数通常较小，因为softmax饱和导致梯度消失
print(Q.grad.norm().item())

# ==================== 第二组实验：使用缩放 ====================
# 意图：展示缩放后梯度的改善

# 重新创建Q和K（因为之前的已经被backward修改）
Q = torch.randn([seq_len, dk], requires_grad=True)
K = torch.randn([seq_len, dk], requires_grad=True)

# 【核心操作】缩放点积注意力
# scores = Q @ K^T / √d_k
# 【原理】假设Q和K的元素独立同分布，均值为0，方差为1
#         则 Q·K^T 的方差 = d_k
#         缩放后方差 = d_k / (√d_k)² = 1
#         这样点积值被控制在合理范围
scores = Q @ K.T / (dk ** 0.5)

# Softmax归一化
# 【改善】缩放后，softmax输入值更小，分布更平滑
#         例如：[2.5, 0.6, 0.4, 0.1] -> softmax -> [0.75, 0.12, 0.09, 0.04]
#         梯度更大，模型更容易学习
weights = torch.softmax(scores, dim=-1)
print(weights)

# 反向传播测试
loss = weights[0,0]
loss.backward()

# 打印Q的梯度范数
# 【观察点】梯度范数通常比不缩放时更大，说明梯度流动更顺畅
print(Q.grad.norm().item())
