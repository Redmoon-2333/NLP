"""
配置文件模块

功能描述:
    集中管理项目的所有配置参数，包括路径配置、模型超参数、训练参数等。
    采用集中式配置便于参数调整和实验管理。

设计原则:
    - 路径配置：使用pathlib保证跨平台兼容性
    - 超参数：参考Transformer原论文设置，可根据硬件调整

作者: Red_Moon
创建日期: 2026-02
"""

from pathlib import Path

# ==================== 路径配置 ====================
# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据目录
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

# ==================== 序列长度配置 ====================
SEQ_LEN = 128
# 意图：平衡内存占用和语义完整性
# 警示：Self-Attention复杂度为O(n²)，增大序列长度会显著增加显存占用

# ==================== 训练超参数 ====================
BATCH_SIZE = 64
# 意图：在显存限制和训练稳定性之间取得平衡
# 警示：批次大小影响梯度估计的方差

DIM_MODEL = 128
# 意图：决定模型的表示能力
# 【与原论文差异】原论文d_model=512，此处为教学目的简化

HIDDEN_SIZE = 256
# 意图：Feed Forward网络的中间层维度
# 【与原论文差异】原论文d_ff=2048，此处按比例缩小

LEARNING_RATE = 1e-3
# 警示：过大导致震荡，过小收敛慢
# 【与原论文差异】原论文使用Noam调度器，此处简化为固定学习率

EPOCHS = 30

# ==================== Transformer架构参数 ====================
NUM_HEADS = 4
# 意图：并行处理不同位置的信息，捕捉不同类型的关系
# 约束：d_model必须能被num_heads整除
# 【与原论文差异】原论文num_heads=8，此处为教学目的简化

NUM_ENCODER_LAYERS = 2
# 【与原论文差异】原论文N=6，此处为教学目的简化

NUM_DECODER_LAYERS = 2
# 【与原论文差异】原论文N=6，此处为教学目的简化
