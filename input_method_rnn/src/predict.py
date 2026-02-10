"""
模型预测模块

功能描述:
    本模块实现了输入法RNN模型的预测功能，主要功能包括：
    1. 批量预测(predict_batch): 对多个输入序列同时进行预测，返回Top-5候选词索引
    2. 单条预测(predict): 对单条文本输入进行预测，返回Top-5候选词
    3. 交互式预测(run_predict): 提供命令行交互界面，实时获取用户输入并给出预测

预测流程:
    1. 文本编码：使用分词器将文本转换为词索引序列
    2. 模型推理：将索引序列输入模型，获取词汇表上的概率分布
    3. Top-K选择：选取概率最高的K个词作为候选结果
    4. 结果解码：将候选词索引转换回文本形式

作者: Red_Moon
创建日期: 2026-02
"""

import jieba
import torch
import config
from model import InputMethodModel
from tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
    """
    批量预测函数

    功能描述:
        对一批输入序列进行预测，返回每个输入的前5个最可能的候选词索引。
        使用torch.topk高效获取Top-K结果，适用于批量评估和推理。

    参数:
        model (InputMethodModel): 已加载权重的输入法模型
        inputs (torch.Tensor): 输入词索引序列
            形状: [batch_size, seq_len]
            类型: torch.long
            设备: 应与模型在同一设备(CPU/GPU)

    返回:
        list: Top-5候选词索引列表
            形状: [batch_size, 5]
            说明: 每个元素是一个包含5个整数索引的列表

    注意事项:
        - 函数内部会调用model.eval()设置评估模式
        - 使用torch.no_grad()上下文管理器禁用梯度计算，节省内存
        - 输入张量应已移至正确的设备(CPU/GPU)
    """
    model.eval()

    with torch.no_grad():
        output = model(inputs)

    top5_indexes = torch.topk(output, k=5).indices
    top5_indexes_list = top5_indexes.tolist()
    return top5_indexes_list


def predict(text, model, tokenizer, device):
    """
    单条文本预测函数

    功能描述:
        对单条文本输入进行预测，返回Top-5候选词。
        完整的预测流程包括：文本编码 -> 张量转换 -> 模型推理 -> 结果解码

    参数:
        text (str): 输入文本字符串，将被分词并编码
        model (InputMethodModel): 已加载权重的输入法模型
        tokenizer (JiebaTokenizer): 分词器，用于文本编码和解码
        device (torch.device): 计算设备（CPU或CUDA）

    返回:
        list: Top-5候选词列表
            形状: [5]
            元素类型: str
            说明: 按概率从高到低排序的候选词列表

    处理流程:
        1. 使用tokenizer将文本编码为词索引列表
        2. 将索引列表转换为张量并添加batch维度
        3. 将张量移至指定设备
        4. 调用predict_batch进行批量预测
        5. 将预测索引解码为词并返回
    """
    indexes = tokenizer.encode(text)
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    input_tensor = input_tensor.to(device)

    top5_indexes_list = predict_batch(model, input_tensor)
    top5_tokens = [tokenizer.index2word[index] for index in top5_indexes_list[0]]

    return top5_tokens


def run_predict():
    """
    运行交互式预测程序

    功能描述:
        提供命令行交互界面，循环接收用户输入并实时显示预测结果。
        支持累积输入历史，模拟真实输入法的连续输入场景。

    交互命令:
        - 输入 'q' 或 'quit': 退出程序
        - 输入空字符串: 提示重新输入
        - 其他输入: 累积到输入历史并进行预测

    资源准备流程:
        1. 确定计算设备（优先使用GPU）
        2. 从文件加载词表
        3. 初始化模型并加载预训练权重

    异常处理:
        - 如果模型文件或词表文件不存在，会抛出FileNotFoundError
        - 如果输入包含词表外的词，会使用<unk>标记处理
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("词表加载成功")

    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))
    print("模型加载成功")

    print("\n" + "=" * 40)
    print("欢迎使用输入法模型(输入q或者quit退出)")
    print("=" * 40)

    input_history = ''

    while True:
        user_input = input("> ")

        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break

        if user_input.strip() == '':
            print("请输入内容")
            continue

        input_history += user_input
        print(f'输入历史: {input_history}')

        try:
            top5_tokens = predict(input_history, model, tokenizer, device)
            print(f'预测结果: {top5_tokens}')
        except Exception as e:
            print(f'预测出错: {e}')

        print("-" * 40)


if __name__ == '__main__':
    run_predict()
