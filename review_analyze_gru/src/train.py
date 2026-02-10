"""
模型训练模块

功能描述:
    本模块实现了基于GRU的情感分析模型的完整训练流程。
    包括：设备配置、数据加载、模型初始化、训练循环、损失计算、
    优化器更新、TensorBoard日志记录以及最佳模型保存。

作者: Red_Moon
创建日期: 2026-02
"""

import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import ReviewAnalyzeModel
from tokenizer import JiebaTokenizer
import config
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个epoch

    功能描述:
        执行一个完整epoch的训练，遍历所有训练批次，
        计算损失并更新模型参数。

    参数:
        model (nn.Module): 待训练的模型
        dataloader (DataLoader): 训练数据加载器
        loss_fn (nn.Module): 损失函数
        optimizer (optim.Optimizer): 优化器
        device (torch.device): 计算设备（CPU或CUDA）

    返回:
        float: 该epoch的平均损失值

    训练流程:
        1. 设置模型为训练模式（启用Dropout等）
        2. 遍历数据加载器，获取输入和目标
        3. 将数据移动到指定设备
        4. 前向传播计算输出
        5. 计算损失
        6. 反向传播计算梯度
        7. 优化器更新参数
        8. 累加损失值

    时间复杂度: O(n)，n为训练样本数量
    空间复杂度: O(batch_size * seq_len * hidden_size)
    """
    model.train()

    total_loss = 0
    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train():
    """
    完整的模型训练流程

    功能描述:
        执行完整的模型训练流程，包括：
        1. 设备配置（优先使用GPU）
        2. 数据加载（训练集）
        3. 分词器初始化（加载词表）
        4. 模型初始化（创建GRU模型）
        5. 损失函数配置（BCEWithLogitsLoss）
        6. 优化器配置（Adam）
        7. TensorBoard日志记录
        8. 多epoch训练循环
        9. 最佳模型保存（基于验证损失）

    训练配置:
        - 设备: CUDA（如果可用）否则CPU
        - 损失函数: BCEWithLogitsLoss（二分类交叉熵）
        - 优化器: Adam（学习率来自config）
        - 训练轮数: config.EPOCHS
        - 早停策略: 保存验证损失最低的模型

    输出:
        - 训练日志输出到控制台
        - TensorBoard日志保存到logs目录
        - 最佳模型权重保存到models/best.pt

    异常处理:
        - 如果词表文件不存在，会抛出FileNotFoundError
        - 如果CUDA设备不可用，自动回退到CPU
    """
    # 1. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2，数据
    dataloader = get_dataloader()
    # 3. 分词器
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")
    # 4. 模型
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size, padding_index=tokenizer.pad_token_index).to(device)
    # 5. 损失函数
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 7. TensorBoard Writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f'======= Epoch {epoch} =======')
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f'loss:{loss:.4f}')
        # 记录Tensorboard
        writer.add_scalar('loss', loss, epoch)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pt')
            print("保存模型")
    writer.close()


if __name__ == '__main__':
    train()
