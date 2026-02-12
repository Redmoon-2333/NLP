"""
模型定义模块

功能描述:
    定义Seq2Seq翻译模型（编码器-解码器架构）。
    使用PyTorch内置的nn.Transformer组件构建完整的翻译模型。

核心组件:
    - PositionEncoding: 位置编码层，为输入注入位置信息
    - TranslationModel: 完整的翻译模型，包含编码器和解码器

设计说明:
    本实现使用PyTorch内置的nn.Transformer，简化了手动实现Multi-Head Attention的复杂度。
    适合初学者理解Transformer的整体架构和数据流向。

作者: Red_Moon
创建日期: 2026-02
"""

import math

import torch
from torch import nn
import config


class PositionEncoding(nn.Module):
    """
    位置编码层
    
    功能:
        为输入序列注入位置信息，弥补Self-Attention的位置无关性。
        Self-Attention本身无法区分不同位置的token，需要显式添加位置编码。
    
    原理:
        使用正弦/余弦函数生成位置编码:
        - PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        其中pos是位置索引，i是维度索引。
    
    优势:
        1. 每个位置有唯一编码
        2. 编码值有界[-1, 1]
        3. 可以外推到训练时未见过的长度
        4. 相对位置可以通过线性变换得到
    
    输入输出:
        - 输入: [batch_size, seq_len, d_model] 词嵌入
        - 输出: [batch_size, seq_len, d_model] 加入位置编码后的嵌入
    """

    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 预计算位置编码矩阵
        # pos.shape: (max_len, 1)
        pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        
        # 维度索引（偶数）
        # _2i.shape: (d_model/2,)
        _2i = torch.arange(0, self.d_model, step=2, dtype=torch.float)
        
        # 计算分母项: 10000^(2i/d_model)
        # 意图：不同维度使用不同的频率，低维度变化快，高维度变化慢
        div_term = torch.pow(10000, _2i / self.d_model)

        # 计算sin和cos值
        sins = torch.sin(pos / div_term)
        coss = torch.cos(pos / div_term)

        # 组合成完整的位置编码矩阵
        # pe.shape: (max_len, d_model)
        pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float)
        pe[:, 0::2] = sins
        pe[:, 1::2] = coss

        # 注册为buffer（不参与梯度计算，但会随模型移动到GPU）
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        # 广播加法：每个batch使用相同的位置编码
        return x + self.pe[:seq_len]


class TranslationModel(nn.Module):
    """
    翻译模型
    
    功能:
        完整的中英翻译模型，基于Transformer架构。
        编码器处理中文输入，解码器生成英文输出。
    
    架构:
        1. 源语言嵌入层（中文）
        2. 目标语言嵌入层（英文）
        3. 位置编码层
        4. Transformer编码器-解码器
        5. 输出线性层
    
    数据流:
        中文输入 -> 嵌入 -> 位置编码 -> 编码器 -> memory
                                              ↓
        英文输入 -> 嵌入 -> 位置编码 -> 解码器 -> 线性层 -> 输出概率
    """
    
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        
        # 中文嵌入层
        # padding_idx: 指定填充标记的索引，该位置的嵌入始终为0
        self.zh_embedding = nn.Embedding(num_embeddings=zh_vocab_size,
                                         embedding_dim=config.DIM_MODEL,
                                         padding_idx=zh_padding_index)
        
        # 英文嵌入层
        self.en_embedding = nn.Embedding(num_embeddings=en_vocab_size,
                                         embedding_dim=config.DIM_MODEL,
                                         padding_idx=en_padding_index)
        
        # 位置编码层（编码器和解码器共享）
        self.position_encoding = PositionEncoding(config.DIM_MODEL, config.DIM_MODEL)

        # Transformer核心
        # batch_first: 输入形状为[batch, seq, feature]而非[seq, batch, feature]
        self.transformer = nn.Transformer(d_model=config.DIM_MODEL,
                                       nhead=config.NUM_HEADS,
                                       num_encoder_layers=config.NUM_ENCODER_LAYERS,
                                       num_decoder_layers=config.NUM_DECODER_LAYERS,
                                       batch_first=True)
        
        # 输出层：将解码器输出映射到词表大小的概率分布
        self.linear = nn.Linear(in_features=config.DIM_MODEL, out_features=en_vocab_size)
    
    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask):
        # 编码阶段
        memory = self.encode(src, src_pad_mask)
        # 解码阶段
        return self.decode(tgt, memory, tgt_pad_mask, src_pad_mask)


    def encode(self, src, src_pad_mask):
        """
        编码阶段
        
        参数:
            src: [batch_size, src_len] 源语言输入
            src_pad_mask: [batch_size, src_len] 源语言填充掩码
                          True表示该位置是<pad>，应被忽略
        
        返回:
            memory: [batch_size, src_len, d_model] 编码器输出
        """
        # 词嵌入
        embed = self.zh_embedding(src)
        # embed.shape: [batch_size, src_len, dim_model]
        
        # 位置编码
        embed = self.position_encoding(embed)

        # 编码器前向传播
        # src_key_padding_mask: 填充掩码，True位置不参与注意力计算
        memory = self.transformer.encoder(src=embed, src_key_padding_mask=src_pad_mask)
        # memory.shape: [batch_size, src_len, d_model]

        return memory


    def decode(self, tgt, memory, tgt_mask, memory_pad_mask):
        """
        解码阶段
        
        参数:
            tgt: [batch_size, tgt_len] 目标语言输入
            memory: [batch_size, src_len, d_model] 编码器输出
            tgt_mask: [tgt_len, tgt_len] 因果掩码（下三角矩阵）
                      防止解码器看到未来信息
            memory_pad_mask: [batch_size, src_len] 编码器输出的填充掩码
        
        返回:
            [batch_size, tgt_len, en_vocab_size] 输出概率分布
        """
        # 词嵌入
        embed = self.en_embedding(tgt)
        
        # 位置编码
        embed = self.position_encoding(embed)
        # embed.shape: [batch_size, tgt_len, dim_model]
        
        # 解码器前向传播
        # tgt_mask: 因果掩码，确保位置i只能看到位置<=i的信息
        decoder_output = self.transformer.decoder(
            tgt=embed, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            memory_key_padding_mask=memory_pad_mask
        )
        
        # 线性变换到词表大小
        output = self.linear(decoder_output)
        return output
