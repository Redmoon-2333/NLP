"""
配置文件模块

功能描述:
    本模块定义了输入法LSTM模型的全局配置参数，包括数据路径、模型超参数等。
    所有路径均基于项目根目录进行定义，确保跨平台兼容性。

作者: Red_Moon
创建日期: 2026-02
"""

from pathlib import Path

# =============================================================================
# 路径配置
# =============================================================================
ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"


# =============================================================================
# 模型超参数配置
# =============================================================================
SEQ_LEN = 128
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 50
