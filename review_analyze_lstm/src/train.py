"""
模型训练模块

功能描述:
    实现基于LSTM的情感分析模型的训练流程。

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
    
    参数:
        model: 待训练模型
        dataloader: 训练数据加载器
        loss_fn: 损失函数
        optimizer: 优化器
        device: 计算设备
    
    返回:
        float: 平均损失
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
    """完整的模型训练流程"""
    # 1. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2. 数据
    dataloader = get_dataloader()
    # 3. 分词器
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")
    # 4. 模型
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size, padding_index=tokenizer.pad_token_index).to(device)
    # 5. 损失函数（警示：BCEWithLogitsLoss内置sigmoid，避免数值不稳定）
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # 6. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 7. TensorBoard
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f'======= Epoch {epoch} =======')
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f'loss:{loss:.4f}')
        writer.add_scalar('loss', loss, epoch)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pt')
            print("保存模型")
    writer.close()


if __name__ == '__main__':
    train()
