"""
数据预处理模块

功能描述:
    本模块实现了机器翻译任务的数据预处理流程。
    主要功能包括：原始平行语料加载、训练/测试集划分、词表构建、
    文本编码和JSONL格式数据保存。

处理流程:
    1. 读取原始平行语料（cmn.txt格式：英文\t中文）
    2. 划分训练集(80%)和测试集(20%)
    3. 基于训练集构建中英文词表
    4. 使用分词器将文本编码为词索引序列
    5. 保存处理后的训练集和测试集为JSONL格式

数据格式:
    输入（cmn.txt）:
        - 每行包含英文句子\t中文句子

    输出（JSONL）:
        每行一个JSON对象，包含:
        - zh: 中文词索引列表
        - en: 英文词索引列表（含<sos>和<eos>标记）

作者: Red_Moon
创建日期: 2026-02
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import EnglishTokenizer, ChineseTokenizer
import config


def process():
    """数据处理主函数"""
    print("开始处理数据")
    
    # 读取原始平行语料
    # cmn.txt格式：英文\t中文
    df = pd.read_csv(
        config.RAW_DATA_DIR / "cmn.txt",
        sep='\t',
        header=None,
        usecols=[0, 1],
        names=["en", "zh"],
        encoding='utf-8'
    ).dropna()
    
    # 划分训练集和测试集
    # 8:2比例是NLP任务的标准做法
    # random_state=42: 固定随机种子，保证结果可复现
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 构建词表
    # 基于训练集构建，避免数据泄露
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(), config.MODELS_DIR / 'zh_vocab.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(), config.MODELS_DIR / 'en_vocab.txt')
    
    # 加载词表
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    
    # 编码训练集
    # 中文不加特殊标记，英文加<sos>和<eos>标记
    # 中文是编码器输入，不需要边界标记
    # 英文是解码器输入/输出，需要边界标记
    train_df['zh'] = train_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    train_df['en'] = train_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))
    train_df.to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)
    
    # 编码测试集
    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, add_sos_eos=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))
    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)
    
    print("数据处理完成")


if __name__ == '__main__':
    process()
