"""
数据集处理模块
本模块定义了Dataset类和DataLoader获取方法，用于加载和处理训练/测试数据
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import config


class InputMethodDataset(Dataset):
    """
    智能输入法数据集类
    
    继承自PyTorch的Dataset类，用于加载和处理jsonl格式的训练数据
    数据格式：每行一个JSON对象，包含'input'（输入序列）和'target'（目标词）字段
    """
    
    def __init__(self, path):
        """
        初始化数据集
        
        Args:
            path (str or Path): 数据文件路径（jsonl格式）
        """
        # 读取jsonl文件，每行是一个JSON对象
        # lines=True表示按行读取，orient="records"表示每行是一个记录
        self.data = pd.read_json(path, lines=True, orient="records").to_dict(orient="records")
    
    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            int: 数据集中样本的数量
        """
        return len(self.data)
    
    def __getitem__(self, index):
        """
        获取指定索引的样本
        
        Args:
            index (int): 样本索引
        
        Returns:
            tuple: (input_tensor, target_tensor)
                - input_tensor: 输入序列张量，形状为 [seq_len]
                - target_tensor: 目标词张量，标量（单个词的索引）
        """
        # 将输入序列转换为Long类型的张量（词索引）
        input_tensor = torch.tensor(self.data[index]['input'], dtype=torch.long)
        # 将目标词转换为Long类型的张量
        target_tensor = torch.tensor(self.data[index]['target'], dtype=torch.long)
        return input_tensor, target_tensor


def get_dataloader(train=True):
    """
    获取DataLoader
    
    根据train参数返回训练集或测试集的DataLoader
    
    Args:
        train (bool): 是否为训练集。True返回训练集，False返回测试集
    
    Returns:
        DataLoader: 配置好的数据加载器
            - batch_size: 从config读取
            - shuffle: True（训练时打乱数据顺序）
    """
    # 根据train参数选择数据文件路径
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    
    # 创建数据集实例
    dataset = InputMethodDataset(path)
    
    # 创建并返回DataLoader
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,  # 批次大小
        shuffle=True                     # 打乱数据顺序
    )


if __name__ == '__main__':
    # ==================== 测试代码 ====================
    # 获取训练集DataLoader
    train_dataset = get_dataloader()
    # 获取测试集DataLoader
    train_dataloader = get_dataloader(train=False)
    
    # 打印数据集大小
    print(f"训练集批次数量: {len(train_dataset)}")
    print(f"测试集批次数量: {len(train_dataloader)}")
    
    # 获取一个批次的数据并打印形状
    for input_tensor, target_tensor in train_dataset:
        print(f"输入张量形状: {input_tensor.shape}")   # 期望: [batch_size, seq_len]
        print(f"目标张量形状: {target_tensor.shape}")  # 期望: [batch_size]
        break
