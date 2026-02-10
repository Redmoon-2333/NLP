"""
模型定义模块

功能描述:
    本模块定义了基于GRU的情感分析模型ReviewAnalyzeModel。
    模型结构：Embedding层 -> GRU层 -> Linear层
    支持变长序列处理，通过提取最后一个有效时间步的隐藏状态进行分类。

作者: Red_Moon
创建日期: 2026-02
"""

import torch.nn as nn
import config
import torch


class ReviewAnalyzeModel(nn.Module):
    """
    基于GRU的评论情感分析模型

    功能描述:
        使用GRU网络对文本序列进行编码，提取序列的语义特征，
        并通过全连接层输出情感分类结果（正向/负向）。

    架构说明:
        1. Embedding层: 将词索引映射为稠密向量表示
        2. GRU层: 建模序列的时序依赖关系，通过重置门和更新门捕获上下文信息
        3. Linear层: 将GRU最终隐藏状态映射到输出空间

    属性:
        embedding (nn.Embedding): 词嵌入层
        GRU (nn.GRU): GRU门控循环单元层
        linear (nn.Linear): 全连接输出层

    处理流程:
        输入词索引序列 -> 词嵌入 -> GRU编码 -> 提取最后时刻隐藏状态 -> 线性变换 -> 输出logits
    """

    def __init__(self, vocab_size, padding_index):
        """
        初始化模型

        参数:
            vocab_size (int): 词表大小，决定Embedding层的输入维度
            padding_index (int): 填充标记<pad>的索引，用于处理变长序列

        初始化说明:
            - Embedding层使用padding_idx参数，确保填充位置不参与梯度计算
            - GRU使用batch_first=True，使输入输出格式为(batch, seq, feature)
            - Linear层输出维度为1，配合BCEWithLogitsLoss进行二分类
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=padding_index)
        self.GRU = nn.GRU(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(config.HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor):
        """
        前向传播

        功能描述:
            将输入的词索引序列通过Embedding、GRU和Linear层，
            输出每个样本的情感预测logits值。

        参数:
            x (torch.Tensor): 输入词索引序列，形状为[batch_size, seq_len]
                              其中每个元素是词表中的索引值

        返回:
            torch.Tensor: 情感预测logits，形状为[batch_size]
                          正值表示正向情感，负值表示负向情感

        实现细节:
            1. 通过Embedding层将词索引转换为稠密向量
            2. GRU处理序列，输出所有时间步的隐藏状态
            3. 计算每个序列的实际长度（排除padding）
            4. 提取每个序列最后一个有效时间步的隐藏状态
            5. 通过Linear层映射到输出空间并压缩维度

        形状变换:
            - 输入 x: [batch_size, seq_len]
            - embed: [batch_size, seq_len, embedding_dim]
            - gru_out: [batch_size, seq_len, hidden_size]
            - last_hidden: [batch_size, hidden_size]
            - out: [batch_size]
        """
        # x.shape : [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape : [batch_size, seq_len, embedding_dim]
        gru_out,_ = self.GRU(embed)
        # gru_out.shape : [batch_size, seq_len, hidden_size]
        batch_indexes = torch.arange(0, gru_out.shape[0])
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        last_hidden = gru_out[batch_indexes, lengths - 1]
        # last_hidden.shape : [batch_size, hidden_size]
        out = self.linear(last_hidden).squeeze(-1)
        # out.shape : [batch_size]
        return out
