"""
数据集模块

作者: Red_Moon
创建日期: 2026-02
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class TranslationDataset(Dataset):
    """翻译数据集类"""
    
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
        获取指定索引样本
        
        返回:
            (input_tensor, target_tensor): 输入和目标张量
        """
        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)
        return input_tensor, target_tensor


def pad_collate(batch):
    """
    填充批次数据（意图：将变长序列填充到相同长度，便于批处理）
    
    参数:
        batch: [(input_tensor, target_tensor), ...]
    
    返回:
        (input_tensor, target_tensor): 填充后的批次张量
    """
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]
    
    # 使用pad_sequence自动填充（padding_value=0对应<pad>标记）
    input_tensor = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=True, padding_value=0)
    target_tensor = torch.nn.utils.rnn.pad_sequence(target_tensors, batch_first=True, padding_value=0)
    
    return input_tensor, target_tensor


def get_dataloader(train=True):
    """
    获取数据加载器
    
    参数:
        train: True为训练集，False为测试集
    
    返回:
        DataLoader: PyTorch数据加载器
    """
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = TranslationDataset(path)
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
