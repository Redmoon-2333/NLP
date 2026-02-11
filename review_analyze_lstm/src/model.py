"""
模型定义模块

功能描述:
    定义基于LSTM的情感分析模型。
    结构：Embedding -> LSTM -> Linear

作者: Red_Moon
创建日期: 2026-02
"""

import torch.nn as nn
import config
import torch


class ReviewAnalyzeModel(nn.Module):
    """
    基于LSTM的评论情感分析模型
    
    处理流程: 词索引序列 -> Embedding -> LSTM -> 最后时刻隐藏状态 -> Linear -> logits
    """

    def __init__(self, vocab_size, padding_index):
        """
        初始化模型
        
        参数:
            vocab_size: 词表大小
            padding_index: 填充标记<pad>的索引（警示：确保填充位置不参与梯度计算）
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=padding_index)
        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(config.HIDDEN_SIZE, 1)

    def forward(self, x: torch.Tensor):
        """
        前向传播
        
        参数:
            x: 输入词索引序列 [batch_size, seq_len]
        
        返回:
            情感预测logits [batch_size]，正值表示正向，负值表示负向
        
        形状变换:
            x: [batch, seq_len] -> embed: [batch, seq_len, embedding_dim] -> 
            lstm_out: [batch, seq_len, hidden_size] -> last_hidden: [batch, hidden_size] -> 
            out: [batch]
        """
        embed = self.embedding(x)
        lstm_out, (_, _) = self.lstm(embed)
        # 提取每个序列最后一个有效时间步（意图：处理变长序列，取真实最后时刻）
        batch_indexes = torch.arange(0, lstm_out.shape[0])
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        last_hidden = lstm_out[batch_indexes, lengths - 1]
        out = self.linear(last_hidden).squeeze(-1)
        return out
