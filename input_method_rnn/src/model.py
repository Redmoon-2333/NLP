"""
神经网络模型模块

功能描述:
    本模块定义了基于RNN的输入法预测神经网络模型。
    模型架构采用经典的Embedding + RNN + Linear三层结构：
    1. 嵌入层(Embedding): 将词索引转换为密集向量表示
    2. RNN层: 处理序列信息，捕捉上下文依赖
    3. 全连接层(Linear): 将RNN输出映射到词汇表空间，生成预测概率

模型特点:
    - 使用PyTorch nn.Module实现，支持GPU加速
    - 采用batch_first=True，便于数据处理
    - 使用最后一个时间步的隐藏状态进行预测

作者: Red_Moon
创建日期: 2026-02
"""

from torch import nn
import config


class InputMethodModel(nn.Module):
    """
    输入法预测神经网络模型

    功能描述:
        基于RNN的序列预测模型，用于根据输入的前N个词预测下一个词。
        模型接收固定长度的词索引序列，输出词汇表上每个词的概率分布。

    架构说明:
        1. Embedding层: 将离散词索引映射为连续向量
        2. RNN层: 处理序列，提取时序特征
        3. Linear层: 将RNN输出映射到词汇表维度

    属性:
        embedding (nn.Embedding): 词嵌入层
        rnn (nn.RNN): 循环神经网络层
        linear (nn.Linear): 全连接输出层
    """

    def __init__(self, vocab_size):
        """
        初始化输入法模型

        参数:
            vocab_size (int): 词汇表大小，决定嵌入层和输出层的维度
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

        功能描述:
            执行模型的前向传播计算，将输入的词索引序列转换为词汇表上的概率分数。
            使用RNN最后一个时间步的隐藏状态进行预测。

        参数:
            x (torch.Tensor): 输入词索引序列
                形状: [batch_size, seq_len]
                类型: torch.long
                取值范围: [0, vocab_size-1]

        返回:
            torch.Tensor: 词汇表上的预测分数（logits）
                形状: [batch_size, vocab_size]
                说明: 未经过softmax，可直接用于CrossEntropyLoss

        计算流程:
            1. 嵌入层: [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
            2. RNN层: [batch_size, seq_len, embedding_dim] -> [batch_size, seq_len, hidden_size]
            3. 取最后一个时间步: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
            4. 全连接层: [batch_size, hidden_size] -> [batch_size, vocab_size]
        """
        embed = self.embedding(x)
        output, _ = self.rnn(embed)
        last_hidden_state = output[:, -1, :]
        output = self.linear(last_hidden_state)
        return output
