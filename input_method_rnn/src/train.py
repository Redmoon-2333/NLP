"""
模型训练模块
本模块负责模型的训练流程，包括数据加载、模型初始化、训练循环、验证和保存
"""
from pathlib import Path
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import InputMethodModel


def train():
    """
    模型训练主函数
    
    执行完整的训练流程：
    1. 设备选择（GPU/CPU）
    2. 数据加载
    3. 模型初始化
    4. 损失函数和优化器设置
    5. 训练循环
    6. 模型保存
    """
    # ==================== Step 1: 设备选择 ====================
    # 优先使用GPU（cuda），否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ==================== Step 2: 数据加载 ====================
    # 获取训练集和测试集的DataLoader
    train_dataloader = get_dataloader(train=True)   # 训练集
    test_dataloader = get_dataloader(train=False)   # 测试集

    # ==================== Step 3: 模型初始化 ====================
    # 从词表文件获取词表大小
    vocab_size = len(open(config.MODELS_DIR / 'vocab.txt', encoding="utf-8").read().split("\n"))
    print(f"词表大小: {vocab_size}")
    
    # 创建模型实例
    model = InputMethodModel(vocab_size).to(device)
    print("模型初始化完成")

    # ==================== Step 4: 损失函数和优化器 ====================
    # 交叉熵损失函数，适用于多分类任务
    criterion = nn.CrossEntropyLoss()
    # Adam优化器，自适应学习率
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # ==================== Step 5: TensorBoard日志 ====================
    # 创建日志目录，使用时间戳命名
    log_dir = config.LOGS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)

    # ==================== Step 6: 训练循环 ====================
    # 记录最佳验证损失，用于保存最优模型
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        # -------------------- 训练阶段 --------------------
        model.train()  # 设置为训练模式（启用dropout等）
        train_loss = 0.0
        
        # 使用tqdm显示训练进度
        for input_tensor, target_tensor in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            # 将数据移动到指定设备（GPU/CPU）
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            # 前向传播：获取模型预测结果
            output = model(input_tensor)
            
            # 计算损失
            loss = criterion(output, target_tensor)

            # 反向传播三部曲
            optimizer.zero_grad()   # 1. 清空梯度
            loss.backward()         # 2. 计算梯度
            optimizer.step()        # 3. 更新参数

            # 累加损失（用于计算平均损失）
            train_loss += loss.item()

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_dataloader)
        
        # -------------------- 验证阶段 --------------------
        model.eval()  # 设置为评估模式（禁用dropout等）
        val_loss = 0.0
        
        # 验证时不需要计算梯度，使用torch.no_grad()加速
        with torch.no_grad():
            for input_tensor, target_tensor in test_dataloader:
                input_tensor = input_tensor.to(device)
                target_tensor = target_tensor.to(device)

                # 前向传播
                output = model(input_tensor)
                # 计算损失
                loss = criterion(output, target_tensor)
                val_loss += loss.item()

        # 计算平均验证损失
        avg_val_loss = val_loss / len(test_dataloader)

        # -------------------- 日志记录 --------------------
        print(f"Epoch {epoch + 1}/{config.EPOCHS} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # 写入TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)

        # -------------------- 模型保存 --------------------
        # 如果当前验证损失更低，保存为最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pth')
            print(f"保存最佳模型，验证损失: {avg_val_loss:.4f}")

    # 关闭TensorBoard writer
    writer.close()
    print("训练完成")


if __name__ == '__main__':
    train()
