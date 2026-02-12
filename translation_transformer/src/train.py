"""
模型训练模块

功能描述:
    实现Transformer翻译模型的完整训练流程。
    包括数据加载、模型初始化、训练循环、模型保存等功能。

核心组件:
    - train_one_epoch: 单轮训练函数
    - train: 完整训练流程

训练策略:
    - Teacher Forcing: 使用真实标签作为解码器输入
    - 交叉熵损失: 忽略<pad>标记的损失
    - Adam优化器: 自适应学习率

作者: Red_Moon
创建日期: 2026-02
"""

import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from model import TranslationModel
import config
from tokenizer import ChineseTokenizer, EnglishTokenizer


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个轮次
    
    参数:
        model: 待训练模型
        dataloader: 训练数据加载器
        loss_fn: 损失函数（交叉熵）
        optimizer: 优化器（Adam）
        device: 计算设备（CPU/GPU）
    
    返回:
        float: 平均损失
    """
    model.train()
    total_loss = 0
    
    for inputs, targets in tqdm(dataloader, desc='训练'):
        encoder_inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Teacher Forcing策略
        # decoder_inputs: 去掉<eos>，作为解码器输入
        # decoder_targets: 去掉<sos>，作为预测目标
        decoder_inputs = targets[:, :-1]
        decoder_targets = targets[:, 1:]
        
        # 源语言填充掩码
        # True表示该位置是<pad>，应被忽略
        src_pad_mask = encoder_inputs == model.zh_embedding.padding_idx
        
        # 目标语言因果掩码
        # generate_square_subsequent_mask生成下三角矩阵
        tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_inputs.shape[1]).to(device)

        # 前向传播
        decoder_outputs = model(encoder_inputs, decoder_inputs, src_pad_mask, tgt_mask)

        # 计算损失
        # reshape: 将批次和序列维度展平
        loss = loss_fn(decoder_outputs.reshape(-1, decoder_outputs.shape[-1]), decoder_targets.reshape(-1))
        total_loss += loss.item()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss / len(dataloader)


def train():
    """执行完整训练流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    dataloader = get_dataloader()
    print(f"训练集批次数量: {len(dataloader)}")
    
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    print(f"中文词表大小: {zh_tokenizer.vocab_size}")
    print(f"英文词表大小: {en_tokenizer.vocab_size}")
    
    model = TranslationModel(
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_index=zh_tokenizer.pad_token_index,
        en_padding_index=en_tokenizer.pad_token_index
    ).to(device)
    print("模型初始化完成")
    
    # 损失函数
    # ignore_index确保<pad>标记不参与损失计算
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # TensorBoard日志
    log_dir = config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard日志目录: {log_dir}")
    
    best_loss = float('inf')
    
    for epoch in range(1, 1 + config.EPOCHS):
        print("\n" + "=" * 10 + f" Epoch: {epoch}/{config.EPOCHS} " + "=" * 10)
        
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"平均损失: {loss:.6f}")
        
        writer.add_scalar('Loss/train', loss, epoch)
        
        # 保存最佳模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pth')
            print(f"模型保存成功（最佳损失: {best_loss:.6f}）")
    
    writer.close()
    print("\n" + "=" * 40)
    print("训练完成！")
    print(f"最佳损失: {best_loss:.6f}")
    print("=" * 40)


if __name__ == '__main__':
    train()
