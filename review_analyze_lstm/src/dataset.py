"""
数据集模块

功能描述:
    定义情感分析LSTM模型的数据集类。

作者: Red_Moon
创建日期: 2026-02
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class ReviewAnalyzeDataset(Dataset):
    """情感分析数据集类"""

    def __init__(self, path):
        """
        初始化数据集
        
        参数:
            path: JSONL格式数据文件路径
        """
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        获取指定索引的样本
        
        返回:
            (input_tensor, target_tensor): 输入序列[seq_len]和目标标签[]
        """
        input_tensor = torch.tensor(self.data[index]['review'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['label'], dtype=torch.float)
        return input_tensor, target_tensor


def get_dataloader(train=True):
    """
    获取数据加载器
    
    参数:
        train: True为训练集，False为测试集
    
    返回:
        DataLoader: 数据加载器，返回(input_tensor[batch,seq_len], target_tensor[batch])
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
