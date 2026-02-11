"""
数据预处理模块

功能描述:
    本模块实现了输入法RNN模型的数据预处理流程。
    主要功能包括：原始对话数据加载、训练/测试集划分、词表构建、
    文本编码和JSONL格式数据保存。

处理流程:
    1. 读取原始对话数据（synthesized_.jsonl）
    2. 划分训练集(80%)和测试集(20%)
    3. 基于训练集构建词表并保存
    4. 使用分词器将文本编码为词索引序列（inputs和targets是滑动窗口关系）
    5. 保存处理后的训练集和测试集为JSONL格式

数据格式:
    输入（JSONL）:
        - sentence: 原始句子文本

    输出（JSONL）:
        每行一个JSON对象，包含:
        - input: 输入词索引列表（前N个词）
        - target: 目标词索引（第N+1个词）

作者: Red_Moon
创建日期: 2026-02
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer
import config


def process():
    """数据处理主函数"""
    print("开始处理数据")
    
    # 读取原始对话数据
    df = pd.read_json(config.RAW_DATA_DIR / "synthesized_.jsonl", lines=True, orient='records')
    
    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 构建词表
    JiebaTokenizer.build_vocab(train_df['sentence'].tolist(), config.MODELS_DIR / 'vocab.txt')
    
    # 加载词表
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    
    # 编码训练集（inputs和targets是滑动窗口关系）
    train_df['input'] = train_df['sentence'].apply(lambda x: tokenizer.encode(x, seq_len=config.SEQ_LEN))
    train_df['target'] = train_df['sentence'].apply(lambda x: tokenizer.encode(x, seq_len=config.SEQ_LEN))
    train_df.to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)
    
    # 编码测试集
    test_df['input'] = test_df['sentence'].apply(lambda x: tokenizer.encode(x, seq_len=config.SEQ_LEN))
    test_df['target'] = test_df['sentence'].apply(lambda x: tokenizer.encode(x, seq_len=config.SEQ_LEN))
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)
    
    print("数据处理完成")


if __name__ == '__main__':
    process()
