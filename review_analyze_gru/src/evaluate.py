"""
模型评估模块

功能描述:
    本模块实现了基于GRU的情感分析模型的评估流程。
    主要功能包括：加载训练好的模型、在测试集上进行预测、
    计算并输出分类准确率。

作者: Red_Moon
创建日期: 2026-02
"""

import torch
import config
from model import ReviewAnalyzeModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer


def evaluate(model, test_dataloader, device):
    """
    评估模型在测试集上的性能

    功能描述:
        在测试集上运行模型预测，计算分类准确率。
        使用0.5作为二分类阈值，将预测概率转换为类别标签。

    参数:
        model (nn.Module): 已加载权重的模型
        test_dataloader (DataLoader): 测试数据加载器
        device (torch.device): 计算设备

    返回:
        float: 分类准确率，范围[0.0, 1.0]

    评估流程:
        1. 设置模型为评估模式
        2. 遍历测试数据批次
        3. 对每个批次进行预测（通过predict_batch）
        4. 应用sigmoid将logits转换为概率
        5. 使用0.5阈值将概率转换为类别（0或1）
        6. 与真实标签比较，统计正确预测数量
        7. 计算并返回准确率

    评估指标:
        - 准确率 = 正确预测数 / 总样本数
        - 阈值: 0.5（概率>0.5预测为正向，否则负向）

    时间复杂度: O(n)，n为测试样本数量
    空间复杂度: O(batch_size * seq_len)
    """
    total_count = 0
    correct_count = 0
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        targets = targets.tolist()
        batch_result = predict_batch(model, inputs)
        for result, target in zip(batch_result, targets):
            result = 1 if result > 0.5 else 0
            if result == target:
                correct_count += 1
            total_count += 1
    return correct_count / total_count


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

    输出:
        在控制台打印评估结果，包括使用的设备、词表加载状态、
        模型加载状态以及最终的分类准确率。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("词表加载成功")

    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size, padding_index=tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print("模型加载成功")

    test_dataloader = get_dataloader(train=False)

    acc = evaluate(model, test_dataloader, device)
    print("\n" + "=" * 40)
    print("评估结果")
    print("=" * 40)
    print(f"准确率: {acc:.4f} ({acc*100:.2f}%)")
    print("=" * 40)


if __name__ == '__main__':
    run_evaluate()
