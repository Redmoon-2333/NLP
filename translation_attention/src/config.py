"""
配置文件模块

作者: Red_Moon
创建日期: 2026-02
"""

from pathlib import Path

# 项目根目录（意图：确保所有路径基于项目根目录，保证跨平台兼容性）
ROOT_DIR = Path(__file__).parent.parent

# 数据目录
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

# 序列长度配置
SEQ_LEN = 128  # 最大序列长度（意图：平衡内存占用和语义完整性）

# 训练超参数
BATCH_SIZE = 64  # 批次大小（意图：在显存限制和训练稳定性之间取得平衡）
EMBEDDING_DIM = 128  # 词嵌入维度
HIDDEN_SIZE = 256  # 隐藏层维度（警示：增大可提升表达能力但会增加计算量）
LEARNING_RATE = 1e-3  # 学习率（警示：过大导致震荡，过小收敛慢）
EPOCHS = 30  # 训练轮数
