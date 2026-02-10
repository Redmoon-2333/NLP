"""
模型评估模块

功能描述:
    本模块实现了输入法RNN模型的评估功能，主要功能包括：
    1. 计算Top-1和Top-5准确率，评估模型预测性能
    2. 加载预训练模型和词表，在测试集上进行评估
    3. 提供完整的评估流程，包括资源准备、模型加载、指标计算

评估指标说明:
    - Top-1准确率：模型预测的第一个候选词与真实标签匹配的比例
    - Top-5准确率：真实标签出现在模型预测的前5个候选词中的比例

作者: Red_Moon
创建日期: 2026-02
"""

import torch
import config
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer


def evaluate(model, test_dataloader, device):
    """
    评估模型在测试集上的性能

    功能描述:
        遍历测试集，计算模型的Top-1和Top-5准确率。
        对于每个批次，获取模型预测的前5个候选词索引，
        然后统计真实标签与预测结果的匹配情况。

    参数:
        model (InputMethodModel): 待评估的输入法模型
        test_dataloader (DataLoader): 测试集数据加载器
        device (torch.device): 计算设备（CPU或CUDA）

    返回:
        tuple: (top1_acc, top5_acc)
            - top1_acc (float): Top-1准确率，范围[0, 1]
            - top5_acc (float): Top-5准确率，范围[0, 1]
    """
    top1_acc_count = 0
    top5_acc_count = 0
    total_count = 0

    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        targets = targets.tolist()
        top5_indexes_list = predict_batch(model, inputs)

        for target, top5_indexes in zip(targets, top5_indexes_list):
            total_count += 1
            if target == top5_indexes[0]:
                top1_acc_count += 1
            if target in top5_indexes:
                top5_acc_count += 1

    top1_acc = top1_acc_count / total_count
    top5_acc = top5_acc_count / total_count
    return top1_acc, top5_acc


def run_evaluate():
    """
    运行完整的评估流程

    功能描述:
        执行完整的模型评估流程，包括：
        1. 确定计算设备（优先使用GPU）
        2. 加载词表文件
        3. 初始化模型并加载预训练权重
        4. 加载测试集数据
        5. 执行评估并输出结果

    异常处理:
        如果模型文件或词表文件不存在，会抛出FileNotFoundError
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("词表加载成功")

    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))
    print("模型加载成功")

    test_dataloader = get_dataloader(train=False)

    top1_acc, top5_acc = evaluate(model, test_dataloader, device)
    print("\n" + "=" * 40)
    print("评估结果")
    print("=" * 40)
    print(f"Top-1 准确率: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"Top-5 准确率: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print("=" * 40)


if __name__ == '__main__':
    run_evaluate()
