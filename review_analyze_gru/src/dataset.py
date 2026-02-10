"""
数据集模块

功能描述:
    本模块定义了情感分析GRU模型的数据集类和相关工具函数。
    主要功能包括：
    1. 自定义Dataset类，用于加载和索引预处理后的JSONL格式数据
    2. 提供获取DataLoader的便捷方法，支持训练集和测试集的加载
    3. 处理输入序列和目标标签的张量转换

作者: Red_Moon
创建日期: 2026-02
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class ReviewAnalyzeDataset(Dataset):
    """
    情感分析数据集类

    功能描述:
        继承自PyTorch的Dataset类，用于加载和索引预处理后的训练/测试数据。
        数据格式为JSONL，每行包含一个样本，具有'input'和'target'字段。

    属性:
        data (list): 从JSONL文件加载的数据列表，每个元素是一个字典
    """

    def __init__(self, path):
        """
        初始化数据集

        参数:
            path (str or Path): JSONL格式数据文件的路径
        """
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        """
        获取数据集大小

        返回:
            int: 数据集中样本的总数量
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        获取指定索引的样本

        参数:
            index (int): 样本索引

        返回:
            tuple: (input_tensor, target_tensor)
                - input_tensor (torch.Tensor): 输入序列张量，形状为[seq_len]，dtype为long
                - target_tensor (torch.Tensor): 目标标签张量，形状为[]，dtype为long
        """
        input_tensor = torch.tensor(self.data[index]['review'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['label'], dtype=torch.float)
        return input_tensor, target_tensor


def get_dataloader(train=True):
    """
    获取数据加载器

    功能描述:
        根据train参数自动选择训练集或测试集，创建对应的DataLoader。
        训练集启用shuffle以打乱数据顺序，测试集保持顺序不变。

    参数:
        train (bool, optional): 是否为训练集。默认为True。
            - True: 加载训练集 (train.jsonl)
            - False: 加载测试集 (test.jsonl)

    返回:
        DataLoader: PyTorch数据加载器，每次迭代返回一个批次的(input_tensor, target_tensor)
            - input_tensor形状: [batch_size, seq_len]
            - target_tensor形状: [batch_size]
    """
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = ReviewAnalyzeDataset(path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for input_tensor, target_tensor in train_dataloader:
        print(input_tensor.shape)
        print(target_tensor.shape)
        break
