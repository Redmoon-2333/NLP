"""
数据预处理模块

功能描述:
    本模块实现了情感分析任务的数据预处理流程。
    主要功能包括：原始数据加载、训练/测试集划分、词表构建、
    文本编码和JSONL格式数据保存。

作者: Red_Moon
创建日期: 2026-02
"""

from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer
import config
import pandas as pd


def process():
    """
    执行完整的数据预处理流程

    功能描述:
        将原始CSV格式的评论数据转换为模型可训练的JSONL格式。
        处理流程包括数据加载、清洗、划分、词表构建、文本编码和保存。

    处理流程:
        1. 读取原始CSV数据（online_shopping_10_cats.csv）
        2. 选择label和review列，删除缺失值
        3. 采样10%数据用于快速实验（可根据需要调整）
        4. 按标签分层划分训练集(80%)和测试集(20%)
        5. 基于训练集构建词表并保存
        6. 使用分词器将文本编码为词索引序列
        7. 保存处理后的训练集和测试集为JSONL格式

    数据格式:
        输入（CSV）:
            - label: 0（负向）或1（正向）
            - review: 评论文本内容

        输出（JSONL）:
            每行一个JSON对象，包含:
            - review: 词索引列表（编码后的序列）
            - label: 情感标签（0或1）

    编码说明:
        - 使用JiebaTokenizer进行中文分词
        - 序列长度由config.SEQ_LEN控制
        - 自动进行截断或填充

    异常处理:
        - 如果原始数据文件不存在，会抛出FileNotFoundError
        - 如果编码不是UTF-8，可能需要调整encoding参数

    输出:
        - models/vocab.txt: 构建的词表文件
        - data/processed/train.jsonl: 处理后的训练集
        - data/processed/test.jsonl: 处理后的测试集
    """
    print("开始处理数据")
    df = pd.read_csv(config.RAW_DATA_DIR / "online_shopping_10_cats.csv", usecols=["label", "review"], encoding="utf-8").dropna().sample(frac=0.1)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"])

    JiebaTokenizer.build_vocab(train_df['review'].tolist(), config.MODELS_DIR / 'vocab.txt')

    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')

    # # 计算序列长度
    # print(train_df['review'].apply(lambda x:len(tokenizer.tokenize(x))).quantile(0.95))

    train_df['review'] = train_df['review'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))

    train_df.to_json(config.PROCESSED_DATA_DIR / 'train.jsonl', orient='records', lines=True)

    test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))

    test_df.to_json(config.PROCESSED_DATA_DIR / 'test.jsonl', orient='records', lines=True)
    print("数据处理结束")


if __name__ == '__main__':
    process()
