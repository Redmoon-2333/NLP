"""
模型定义模块

功能描述:
    定义Seq2Seq翻译模型（编码器-解码器架构）。

作者: Red_Moon
创建日期: 2026-02
"""

import torch
from torch import nn
import config
class Attention(nn.Module):
    """注意力机制"""
    def forward(self, decoder_hidden, encoder_outputs):
        attention_scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = torch.bmm(attention_weights, encoder_outputs)
        return context_vector

class TranslationEncoder(nn.Module):
    """翻译编码器（意图：将源语言序列编码为上下文向量）"""
    
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_index  # 警示：padding_idx确保<pad>标记不参与梯度计算
        )
        self.GRU = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len]
        
        返回:
            最后时刻隐藏状态 [batch_size, hidden_size]
        """
        embed = self.embedding(x)  # [batch, seq_len, embedding_dim]
        gru_out, _ = self.GRU(embed)  # [batch, seq_len, hidden_size]
        # 提取每个序列最后一个有效时间步（意图：处理变长序列）
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        last_hidden_state = gru_out[torch.arange(gru_out.shape[0]), lengths - 1]
        return gru_out,last_hidden_state  # [batch, hidden_size]


class TranslationDecoder(nn.Module):
    """翻译解码器（意图：根据上下文向量自回归生成目标语言序列）"""
    
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_index
        )
        self.GRU = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True
        )
        self.attention = Attention()
        # 注意：拼接gru_out和context_vector后维度是2*HIDDEN_SIZE
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE * 2, out_features=vocab_size)
    
    def forward(self, x, hidden_0,encoder_outputs):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len]
            hidden_0: 初始隐藏状态 [1, batch_size, hidden_size]
        
        返回:
            output: 词表分布 [batch_size, seq_len, vocab_size]
            hidden_n: 最终隐藏状态 [1, batch_size, hidden_size]
        """
        embed = self.embedding(x)  # [batch, seq_len, embedding_dim]
        gru_out, hidden_n = self.GRU(embed, hidden_0)  # [batch, seq_len, hidden_size]

        # 应用注意力机制（output，encoder_outputs）
        context_vector = self.attention(gru_out, encoder_outputs) #[batch, 1, hidden_size]
        # 融合信息
        combined=torch.cat([gru_out, context_vector], dim=-1)
        output = self.linear(combined)
        return output, hidden_n


class TranslationModel(nn.Module):
    """Seq2Seq翻译模型（编码器-解码器架构）"""
    
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        self.encoder = TranslationEncoder(zh_vocab_size, padding_index=zh_padding_index)
        self.decoder = TranslationDecoder(en_vocab_size, padding_index=en_padding_index)
