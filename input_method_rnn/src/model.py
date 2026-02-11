"""
神经网络模型模块

功能描述:
    定义基于RNN的输入法预测模型（Embedding + RNN + Linear）。

作者: Red_Moon
创建日期: 2026-02
"""

from torch import nn
import config


class InputMethodModel(nn.Module):
    """
    输入法预测神经网络模型
    
    架构: Embedding -> RNN -> Linear
    """

    def __init__(self, vocab_size):
        """
        初始化模型
        
        参数:
            vocab_size: 词汇表大小
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM
        )
        self.rnn = nn.RNN(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True
        )
        self.linear = nn.Linear(
            in_features=config.HIDDEN_SIZE,
            out_features=vocab_size
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入词索引序列 [batch_size, seq_len]
        
        返回:
            词汇表上的预测分数（logits）[batch_size, vocab_size]
        
        形状变换:
            [batch, seq_len] -> [batch, seq_len, embedding_dim] -> 
            [batch, seq_len, hidden_size] -> [batch, hidden_size] -> [batch, vocab_size]
        """
        embed = self.embedding(x)
        output, _ = self.rnn(embed)
        last_hidden_state = output[:, -1, :]
        output = self.linear(last_hidden_state)
        return output
