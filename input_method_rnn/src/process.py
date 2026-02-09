"""
数据预处理模块
本模块负责将原始对话数据清洗、分词、编码，并划分为训练集和测试集
"""
import jieba
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config


def build_dataset(sentences, word2index):
    """
    构建数据集
    
    将句子列表转换为模型可读取的(input, target)格式
    使用滑动窗口生成训练样本：前SEQ_LEN个词作为输入，下一个词作为目标
    
    Args:
        sentences (list): 句子列表，每个句子是字符串
        word2index (dict): 词到索引的映射字典
    
    Returns:
        list: 数据集列表，每个元素是字典{'input': [...], 'target': int}
              input: 长度为SEQ_LEN的词索引列表
              target: 下一个词的索引
    """
    # 将所有句子分词并转换为索引序列
    # jieba.lcut(sentence)对句子进行分词
    # word2index.get(token, 0)将词转换为索引，未登录词映射为0(<unk>)
    indexed_sentences = [
        [word2index.get(token, 0) for token in jieba.lcut(sentence)]
        for sentence in sentences
    ]

    dataset = []
    # 遍历每个句子，使用滑动窗口生成训练样本
    for sentence in tqdm(indexed_sentences, desc="构建数据集"):
        # 滑动窗口：从句首到句尾-SEQ_LEN的位置
        for i in range(len(sentence) - config.SEQ_LEN):
            # 输入：连续的SEQ_LEN个词
            input_seq = sentence[i:i + config.SEQ_LEN]
            # 目标：输入序列的下一个词
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': input_seq, 'target': target})
    return dataset


def process():
    """
    主处理函数
    
    执行完整的数据预处理流程：
    1. 读取原始数据
    2. 提取句子
    3. 划分训练集/测试集
    4. 构建词表
    5. 保存词表
    6. 构建并保存训练集
    7. 构建并保存测试集
    """
    print("开始处理数据")
    
    # ==================== Step 1: 读取原始数据 ====================
    # 读取jsonl格式的原始对话数据
    # sample(frac=0.1)随机采样10%的数据用于快速实验
    df = pd.read_json(
        config.RAW_DATA_DIR / "synthesized_.jsonl",
        lines=True,
        orient="records"
    ).sample(frac=0.1)
    
    # ==================== Step 2: 提取句子 ====================
    # 从对话数据中提取所有句子
    # 数据格式：每行是一个对话，包含多个user的句子
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            # 句子格式："user1：具体内容"，使用split("：")提取内容部分
            sentences.append(sentence.split("：")[1])
    
    print(f"句子总数：{len(sentences)}")

    # ==================== Step 3: 划分数据集 ====================
    # 使用sklearn的train_test_split划分训练集和测试集
    # test_size=0.2表示测试集占20%
    train_sentences, test_sentences = train_test_split(
        sentences,
        test_size=0.2,
        random_state=42  # 随机种子，保证可复现
    )

    # ==================== Step 4: 构建词表 ====================
    # 使用集合去重，收集所有训练集中的词
    vocab_set = set()
    for sentence in tqdm(train_sentences, desc="构建词表"):
        vocab_set.update(jieba.lcut(sentence))
    
    # 词表列表，第一个词是<unk>（未知词标记）
    vocab_list = ["<unk>"] + list(vocab_set)
    print(f"词表大小：{len(vocab_list)}")

    # ==================== Step 5: 保存词表 ====================
    # 将词表保存为txt文件，每行一个词
    with open(config.MODELS_DIR / 'vocab.txt', 'w', encoding="utf-8") as f:
        f.write("\n".join(vocab_list))
    print("词表保存成功")

    # ==================== Step 6: 构建训练集 ====================
    # 创建词到索引的映射字典
    word2index = {word: index for index, word in enumerate(vocab_list)}
    # 构建训练数据集
    train_dataset = build_dataset(train_sentences, word2index)
    print(f"训练集样本数：{len(train_dataset)}")
    
    # 保存训练集为jsonl格式
    pd.DataFrame(train_dataset).to_json(
        config.PROCESSED_DATA_DIR / 'train.jsonl',
        lines=True,
        orient="records"
    )
    print("训练集保存成功")

    # ==================== Step 7: 构建测试集 ====================
    test_dataset = build_dataset(test_sentences, word2index)
    print(f"测试集样本数：{len(test_dataset)}")
    
    # 保存测试集为jsonl格式
    pd.DataFrame(test_dataset).to_json(
        config.PROCESSED_DATA_DIR / 'test.jsonl',
        lines=True,
        orient="records"
    )
    print("测试集保存成功")

    print("数据预处理完成")


if __name__ == "__main__":
    process()
