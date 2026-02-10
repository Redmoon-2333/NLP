"""
模型预测模块

功能描述:
    本模块实现了基于LSTM的情感分析模型的预测功能。
    支持批量预测和单条文本预测，提供交互式命令行界面。

作者: Red_Moon
创建日期: 2026-02
"""

import torch
import config
from model import ReviewAnalyzeModel
from tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
    """
    批量预测

    功能描述:
        对输入批次进行情感预测，返回每个样本属于正向情感的概率。
        使用sigmoid函数将模型输出的logits转换为概率值。

    参数:
        model (nn.Module): 已加载权重的模型
        inputs (torch.Tensor): 输入张量，形状为[batch_size, seq_len]

    返回:
        list: 预测概率列表，每个元素是[0.0, 1.0]范围内的float值
              值越接近1表示正向情感概率越高，越接近0表示负向情感概率越高

    实现细节:
        1. 设置模型为评估模式（禁用Dropout等）
        2. 使用torch.no_grad()禁用梯度计算，节省内存
        3. 模型前向传播获取logits输出
        4. 应用sigmoid激活函数将logits映射到(0,1)概率空间
        5. 转换为Python列表返回

    时间复杂度: O(batch_size * seq_len * hidden_size)
    空间复杂度: O(batch_size * hidden_size)
    """
    model.eval()

    with torch.no_grad():
        output = model(inputs)
    batch_result = torch.sigmoid(output)
    return batch_result.tolist()


def predict(text, model, tokenizer, device):
    """
    单条文本预测

    功能描述:
        对单条文本进行情感分析预测，返回该文本属于正向情感的概率。

    参数:
        text (str): 待预测的文本字符串
        model (nn.Module): 已加载权重的模型
        tokenizer (JiebaTokenizer): 分词器实例
        device (torch.device): 计算设备

    返回:
        float: 正向情感预测概率，范围[0.0, 1.0]

    处理流程:
        1. 使用分词器将文本编码为词索引序列
        2. 将索引序列转换为张量并添加批次维度
        3. 将张量移动到指定设备
        4. 调用predict_batch进行批量预测（单条作为batch_size=1）
        5. 返回预测结果

    编码说明:
        - 使用tokenizer.encode进行分词和编码
        - 序列长度由config.SEQ_LEN控制
        - 自动进行截断或填充
    """
    indexes = tokenizer.encode(text, seq_len=config.SEQ_LEN)
    input_tensor = torch.tensor([indexes], dtype=torch.long)
    input_tensor = input_tensor.to(device)

    batch_result = predict_batch(model, input_tensor)

    return batch_result[0]


def run_predict():
    """
    运行交互式预测界面

    功能描述:
        启动交互式命令行界面，允许用户输入文本并获取情感分析结果。
        支持连续多轮预测，直到用户输入退出命令。

    交互流程:
        1. 配置计算设备
        2. 加载词表和分词器
        3. 初始化模型并加载预训练权重
        4. 进入交互循环，等待用户输入
        5. 对输入文本进行情感预测
        6. 输出预测结果和情感类别判断
        7. 支持'q'或'quit'命令退出

    输出格式:
        - 显示预测概率值
        - 判断情感类别（正向/负向）
        - 显示置信度（概率值或1-概率值）

    异常处理:
        - 空输入提示重新输入
        - 模型或词表文件不存在时抛出FileNotFoundError
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    jieba_tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print("词表加载成功")

    model = ReviewAnalyzeModel(vocab_size=jieba_tokenizer.vocab_size, padding_index=jieba_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print("模型加载成功")

    print("\n" + "=" * 40)
    print("欢迎使用情感分析模型(输入q或者quit退出)")
    print("=" * 40)

    while True:
        user_input = input("> ")

        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break

        if user_input.strip() == '':
            print("请输入内容")
            continue

        result = predict(user_input, model, jieba_tokenizer, device)
        print(f'预测结果: {result}')
        if result > 0.5:
            print(f"正向评论,置信度:{result}")
        else:
            print(f"负向评论,置信度:{1-result}")

        print("-" * 40)


if __name__ == '__main__':
    run_predict()
