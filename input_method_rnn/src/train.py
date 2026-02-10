"""
模型训练模块

功能描述:
    本模块实现了输入法RNN模型的训练流程，主要功能包括：
    1. 单轮次训练(train_one_epoch): 执行一个epoch的训练，包括前向传播、损失计算、反向传播和参数更新
    2. 完整训练流程(train): 管理整个训练过程，包括资源准备、模型初始化、多轮训练、日志记录和模型保存

训练流程:
    1. 准备计算设备（GPU/CPU）
    2. 加载训练数据集
    3. 加载词表
    4. 初始化模型
    5. 定义损失函数和优化器
    6. 创建TensorBoard日志记录器
    7. 循环训练多个epoch，每轮计算loss并保存最佳模型

作者: Red_Moon
创建日期: 2026-02
"""

import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from model import InputMethodModel
import config
from tokenizer import JiebaTokenizer


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个轮次（one epoch）

    功能描述:
        执行一个完整epoch的训练过程，遍历整个训练数据集，
        对每个批次执行前向传播、损失计算、反向传播和参数更新。

    参数:
        model (InputMethodModel): 待训练的输入法模型
        dataloader (DataLoader): 训练数据加载器
        loss_fn (nn.Module): 损失函数，通常为CrossEntropyLoss
        optimizer (optim.Optimizer): 优化器，通常为Adam
        device (torch.device): 计算设备（CPU或CUDA）

    返回:
        float: 当前epoch的平均损失值（总损失/批次数量）

    训练步骤:
        1. 设置模型为训练模式（启用Dropout等）
        2. 遍历数据加载器，获取输入和目标
        3. 将数据移至指定设备
        4. 前向传播：计算模型输出
        5. 计算损失
        6. 反向传播：计算梯度
        7. 参数更新：执行优化器步骤
        8. 清零梯度：为下一轮做准备
        9. 累加损失值

    注意事项:
        - 使用tqdm显示训练进度条
        - 每次迭代后调用optimizer.zero_grad()清零梯度
        - 损失值使用.item()转换为Python标量
    """
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(dataloader, desc='训练'):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    """
    执行完整的模型训练流程

    功能描述:
        管理整个训练过程，从资源准备到模型保存的完整流程：
        1. 确定计算设备
        2. 加载训练数据
        3. 加载词表
        4. 初始化模型
        5. 定义损失函数和优化器
        6. 创建TensorBoard日志记录器
        7. 多轮训练循环，每轮计算loss并保存最佳模型

    模型保存策略:
        - 保存验证损失最低的模型
        - 模型保存路径：config.MODELS_DIR / 'best.pth'
        - 使用torch.save保存模型状态字典(state_dict)

    日志记录:
        - 使用TensorBoard记录每轮的损失值
        - 日志保存路径：config.LOGS_DIR / 时间戳
        - 可在浏览器中通过tensorboard --logdir=logs查看

    异常处理:
        - 如果词表文件不存在，会抛出FileNotFoundError
        - 如果GPU内存不足，会抛出CUDA out of memory错误
        - 可使用Ctrl+C中断训练，已保存的模型不会丢失
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    dataloader = get_dataloader()
    print(f"训练集批次数量: {len(dataloader)}")

    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print(f"词表大小: {tokenizer.vocab_size}")

    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    print("模型初始化完成")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    log_dir = config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard日志目录: {log_dir}")

    best_loss = float('inf')

    for epoch in range(1, 1 + config.EPOCHS):
        print("\n" + "=" * 10 + f" Epoch: {epoch}/{config.EPOCHS} " + "=" * 10)

        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"平均损失: {loss:.6f}")

        writer.add_scalar('Loss/train', loss, epoch)

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pth')
            print(f"模型保存成功（最佳损失: {best_loss:.6f}）")

    writer.close()
    print("\n" + "=" * 40)
    print("训练完成！")
    print(f"最佳损失: {best_loss:.6f}")
    print("=" * 40)


if __name__ == '__main__':
    train()
