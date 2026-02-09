"""
模型评估模块
本模块负责评估训练好的模型在测试集上的性能
"""
import torch
from torch import nn
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import InputMethodModel


def evaluate():
    """
    模型评估主函数
    
    在测试集上评估模型性能，计算准确率和平均损失
    
    Returns:
        tuple: (accuracy, avg_loss)
            - accuracy (float): Top-1准确率
            - avg_loss (float): 平均交叉熵损失
    """
    # ==================== 设备选择 ====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ==================== 加载词表 ====================
    with open(config.MODELS_DIR / 'vocab.txt', encoding="utf-8") as f:
        vocab_list = f.read().split("\n")
    vocab_size = len(vocab_list)
    print(f"词表大小: {vocab_size}")

    # ==================== 加载模型 ====================
    model = InputMethodModel(vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth', map_location=device))
    model.eval()  # 设置为评估模式
    print("模型加载完成")

    # ==================== 加载测试数据 ====================
    test_dataloader = get_dataloader(train=False)
    print(f"测试集批次数量: {len(test_dataloader)}")

    # ==================== 评估指标 ====================
    criterion = nn.CrossEntropyLoss()  # 损失函数
    total_loss = 0.0                    # 累计损失
    correct = 0                         # 正确预测数
    total = 0                           # 总样本数

    # ==================== 评估循环 ====================
    with torch.no_grad():  # 禁用梯度计算，节省内存
        for input_tensor, target_tensor in tqdm(test_dataloader, desc="评估中"):
            # 将数据移动到设备
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            # 前向传播
            output = model(input_tensor)

            # 计算损失
            loss = criterion(output, target_tensor)
            total_loss += loss.item()

            # 计算准确率
            # output形状: [batch_size, vocab_size]
            # torch.max返回最大值和对应的索引
            _, predicted = torch.max(output, dim=1)
            
            # 统计正确预测数
            correct += (predicted == target_tensor).sum().item()
            total += target_tensor.size(0)

    # ==================== 计算最终指标 ====================
    avg_loss = total_loss / len(test_dataloader)
    accuracy = correct / total

    print(f"\n评估结果:")
    print(f"  平均损失: {avg_loss:.4f}")
    print(f"  Top-1 准确率: {accuracy:.4f} ({correct}/{total})")

    return accuracy, avg_loss


if __name__ == '__main__':
    evaluate()
