"""
配置文件
本模块定义了智能输入法RNN模型的所有配置参数和路径设置
"""
from pathlib import Path

# ==================== 路径配置 ====================
# 项目根目录：当前文件的父目录的父目录（即input_method_rnn目录）
ROOT_DIR = Path(__file__).parent.parent

# 原始数据目录：存放未处理的原始对话数据
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"

# 处理后数据目录：存放清洗、分词、编码后的训练/测试数据
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

# 日志目录：存放TensorBoard训练日志
LOGS_DIR = ROOT_DIR / "logs"

# 模型保存目录：存放训练好的模型权重和词表
MODELS_DIR = ROOT_DIR / "models"

# ==================== 模型超参数配置 ====================
# 序列长度：输入模型的token数量（即使用前SEQ_LEN个词预测下一个词）
SEQ_LEN = 5

# 批次大小：每个训练批次包含的样本数
BATCH_SIZE = 64

# 词嵌入维度：每个词被映射到的向量维度
EMBEDDING_DIM = 128

# 隐藏层维度：RNN隐藏状态的大小
HIDDEN_SIZE = 256

# 学习率：优化器的学习率
LEARNING_RATE = 1e-3

# 训练轮数：整个训练数据集被遍历的次数
EPOCHS = 10
