"""
数据预处理模块

功能描述:
    本模块实现了输入法RNN模型的数据预处理流程，主要功能包括：
    1. 读取原始对话数据（JSONL格式）
    2. 提取和清洗句子
    3. 划分训练集和测试集
    4. 构建词表
    5. 构建序列预测数据集（滑动窗口生成input-target对）
    6. 保存处理后的数据为JSONL格式

数据处理流程:
    原始数据 -> 提取句子 -> 划分数据集 -> 构建词表 -> 序列化 -> 保存

作者: Red_Moon
创建日期: 2026-02
"""

import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tokenizer import JiebaTokenizer
import config


def build_dataset(sentences, tokenizer):
    """
    构建序列预测数据集

    功能描述:
        使用滑动窗口方法将句子列表转换为(input, target)格式的训练样本。
        对于每个句子，从左到右滑动窗口，窗口内的词作为输入，下一个词作为预测目标。

    参数:
        sentences (list): 句子列表，每个句子是字符串
        tokenizer (JiebaTokenizer): 分词器，用于将句子编码为词索引序列

    返回:
        list: 数据集列表，每个元素是一个字典，格式为{'input': [...], 'target': int}
            - input: 长度为SEQ_LEN的词索引列表
            - target: 单个词索引（下一个词）

    算法说明:
        对于句子 [w1, w2, w3, w4, w5, w6, w7] 和 SEQ_LEN=5:
        - 样本1: input=[w1,w2,w3,w4,w5], target=w6
        - 样本2: input=[w2,w3,w4,w5,w6], target=w7
    """
    indexed_sentences = [tokenizer.encode(sentence) for sentence in sentences]

    dataset = []

    for sentence in tqdm(indexed_sentences, desc="构建数据集"):
        for i in range(len(sentence) - config.SEQ_LEN):
            input_seq = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': input_seq, 'target': target})

    return dataset


def process():
    """
    执行完整的数据预处理流程

    功能描述:
        执行从原始数据到训练/测试集的完整预处理流程，包括：
        1. 读取原始JSONL数据文件
        2. 从对话中提取句子
        3. 划分训练集和测试集（80/20分割）
        4. 基于训练集构建词表
        5. 构建训练集和测试集的序列样本
        6. 保存处理后的数据到JSONL文件

    数据格式说明:
        原始数据格式（JSONL）:
            {"dialog": ["角色A：句子1", "角色B：句子2", ...]}

        处理后数据格式（JSONL）:
            {"input": [1,2,3,4,5], "target": 6}

    异常处理:
        - 如果原始数据文件不存在，会抛出FileNotFoundError
        - 如果MODELS_DIR目录不存在，需要手动创建
    """
    print("开始处理数据")

    df = pd.read_json(
        config.RAW_DATA_DIR / "synthesized_.jsonl",
        lines=True,
        orient="records"
    ).sample(frac=0.01)

    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            content = sentence.split('：')[1]
            sentences.append(content)

    print(f'句子总数: {len(sentences)}')

    train_sentences, test_sentences = train_test_split(
        sentences,
        test_size=0.2
    )
    print(f'训练集句子数: {len(train_sentences)}')
    print(f'测试集句子数: {len(test_sentences)}')

    JiebaTokenizer.build_vocab(train_sentences, config.MODELS_DIR / 'vocab.txt')

    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')

    train_dataset = build_dataset(train_sentences, tokenizer)
    print(f'训练集样本数: {len(train_dataset)}')

    pd.DataFrame(train_dataset).to_json(
        config.PROCESSED_DATA_DIR / 'train.jsonl',
        orient='records',
        lines=True
    )
    print('训练集保存成功')

    test_dataset = build_dataset(test_sentences, tokenizer)
    print(f'测试集样本数: {len(test_dataset)}')

    pd.DataFrame(test_dataset).to_json(
        config.PROCESSED_DATA_DIR / 'test.jsonl',
        orient='records',
        lines=True
    )
    print('测试集保存成功')

    print("数据处理完成")


if __name__ == '__main__':
    process()
