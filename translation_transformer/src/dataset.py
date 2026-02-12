"""
数据集模块

功能描述:
    定义翻译任务的数据集类和数据加载器。
    负责从JSONL文件读取数据、填充变长序列、构建批次数据。

核心组件:
    - TranslationDataset: 翻译数据集类
    - pad_collate: 批次填充函数
    - get_dataloader: 数据加载器工厂函数

数据格式:
    输入(JSONL): 每行一个JSON对象
    {
        "zh": [中文词索引列表],
        "en": [英文词索引列表]
    }

作者: Red_Moon
创建日期: 2026-02
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class TranslationDataset(Dataset):
    """
    翻译数据集类
    
    功能:
        从JSONL格式文件加载翻译数据对（中文-英文）
    """
    
    def __init__(self, path):
        # 读取JSONL文件（lines=True表示每行一个JSON对象）
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # 从字典中获取词索引列表，转换为LongTensor
        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)
        return input_tensor, target_tensor


def pad_collate(batch):
    """
    批次填充函数（collate_fn）
    
    功能:
        将变长序列填充到批次内最大长度，便于批处理
    
    参数:
        batch: [(input_tensor, target_tensor), ...]
    
    返回:
        tuple: (input_tensor, target_tensor)
            - input_tensor: [batch_size, max_src_len]
            - target_tensor: [batch_size, max_tgt_len]
    """
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]
    
    # 使用pad_sequence自动填充
    # batch_first=True: 输出形状为[batch, seq]
    # padding_value=0: 使用0填充（对应<pad>标记）
    input_tensor = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensor = torch.nn.utils.rnn.pad_sequence(target_tensors, batch_first=True, padding_value=0)
    
    return input_tensor, target_tensor


def get_dataloader(train=True):
    """
    数据加载器工厂函数
    
    参数:
        train: True为训练集，False为测试集
    
    返回:
        DataLoader: PyTorch数据加载器
    """
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = TranslationDataset(path)
    # 训练集打乱，测试集不打乱
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=train, collate_fn=pad_collate)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(f"训练集批次数量: {len(train_dataloader)}")
    print(f"测试集批次数量: {len(test_dataloader)}")
    
    for input_tensor, target_tensor in train_dataloader:
        print(f"输入形状: {input_tensor.shape}")
        print(f"目标形状: {target_tensor.shape}")
        break
