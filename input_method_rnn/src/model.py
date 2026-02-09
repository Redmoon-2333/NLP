"""
模型定义模块
本模块定义了基于RNN的智能输入法模型架构
"""
from torch import nn
import config


class InputMethodModel(nn.Module):
    """
    智能输入法RNN模型
    
    模型结构：Embedding层 -> RNN层 -> Linear层
    输入：前SEQ_LEN个词的索引序列
    输出：下一个词的预测概率分布（vocab_size维）
    """
    
    def __init__(self, vocab_size):
        """
        初始化模型
        
        Args:
            vocab_size (int): 词表大小，即模型需要预测的类别数
        """
        super().__init__()
        
        # ==================== Embedding层 ====================
        # 将词索引映射为稠密向量表示
        # 输入形状: [batch_size, seq_len]（词索引）
        # 输出形状: [batch_size, seq_len, embedding_dim]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,      # 词表大小
            embedding_dim=config.EMBEDDING_DIM  # 嵌入维度
        )

        # ==================== RNN层 ====================
        # 循环神经网络，用于建模序列信息
        # 输入形状: [batch_size, seq_len, embedding_dim]
        # 输出形状: [batch_size, seq_len, hidden_size]
        self.rnn = nn.RNN(
            input_size=config.EMBEDDING_DIM,   # 输入维度（embedding_dim）
            hidden_size=config.HIDDEN_SIZE,    # 隐藏层维度
            batch_first=True                   # 批次维度为第一维
        )

        # ==================== 全连接层 ====================
        # 将RNN的输出映射到词表空间，得到每个词的预测分数
        # 输入形状: [batch_size, hidden_size]
        # 输出形状: [batch_size, vocab_size]
        self.linear = nn.Linear(
            in_features=config.HIDDEN_SIZE,    # 输入维度（RNN隐藏层输出）
            out_features=vocab_size            # 输出维度（词表大小）
        )

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入词索引序列，形状为 [batch_size, seq_len]
        
        Returns:
            torch.Tensor: 预测结果，形状为 [batch_size, vocab_size]
                         表示每个样本对词表中每个词的预测分数
        """
        # x.shape: [batch_size, seq_len]
        
        # Step 1: 词嵌入 - 将词索引转换为向量表示
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim]
        
        # Step 2: RNN处理 - 建模序列信息
        # output: 每个时间步的隐藏状态
        # _: 最后一个时间步的隐藏状态（此处不使用）
        output, _ = self.rnn(embed)
        # output.shape: [batch_size, seq_len, hidden_size]
        
        # Step 3: 取最后一个时间步的输出作为句子表示
        # 使用-1索引获取序列最后一个位置的隐藏状态
        last_hidden_state = output[:, -1, :]
        # last_hidden_state.shape: [batch_size, hidden_size]
        
        # Step 4: 全连接层 - 映射到词表空间
        output = self.linear(last_hidden_state)
        # output.shape: [batch_size, vocab_size]
        
        return output
