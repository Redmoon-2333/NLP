"""
模型训练模块

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
        loss_fn: 损失函数
        optimizer: 优化器
        device: 计算设备
    
    返回:
        float: 平均损失
    """
    model.train()
    total_loss = 0
    
    for inputs, targets in tqdm(dataloader, desc='训练'):
        # 数据移动到设备（意图：确保张量在正确的设备上）
        encoder_inputs = inputs.to(device)  # [batch, src_seq_len]
        targets = targets.to(device)  # [batch, tgt_seq_len]
        
        # 准备解码器输入和目标（意图：Teacher Forcing策略）
        decoder_inputs = targets[:, :-1]  # [batch, tgt_seq_len-1] 去掉<eos>
        decoder_targets = targets[:, 1:]  # [batch, tgt_seq_len-1] 去掉<sos>
        
        # 编码阶段（意图：将源语言编码为上下文向量）
        encoder_outputs,context_vector = model.encoder(encoder_inputs)  # [batch, hidden_size]
        
        # 解码阶段（警示：此处使用循环而非向量化，是为了演示自回归过程）
        decoder_hidden = context_vector.unsqueeze(0)  # [1, batch, hidden_size]
        decoder_outputs = []
        seq_len = decoder_inputs.shape[1]
        
        for i in range(seq_len):
            decoder_input = decoder_inputs[:, i].unsqueeze(1)  # [batch, 1]
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
        
        # 合并输出并reshape（意图：适配CrossEntropyLoss的输入格式）
        decoder_outputs = torch.cat(decoder_outputs, dim=1)  # [batch, seq_len, vocab_size]
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])  # [batch*seq_len, vocab_size]
        decoder_targets = decoder_targets.reshape(-1)  # [batch*seq_len]
        
        # 计算损失并反向传播
        loss = loss_fn(decoder_outputs, decoder_targets)
        total_loss += loss.item()
        
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
    
    # 加载词表（意图：动态获取词表大小，避免硬编码）
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    print(f"中文词表大小: {zh_tokenizer.vocab_size}")
    print(f"英文词表大小: {en_tokenizer.vocab_size}")
    
    # 初始化模型
    model = TranslationModel(
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_index=zh_tokenizer.pad_token_index,
        en_padding_index=en_tokenizer.pad_token_index
    ).to(device)
    print("模型初始化完成")
    
    # 损失函数和优化器（警示：ignore_index确保<pad>标记不参与损失计算）
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # TensorBoard日志（意图：可视化训练过程）
    log_dir = config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard日志目录: {log_dir}")
    
    best_loss = float('inf')
    
    for epoch in range(1, 1 + config.EPOCHS):
        print("\n" + "=" * 10 + f" Epoch: {epoch}/{config.EPOCHS} " + "=" * 10)
        
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"平均损失: {loss:.6f}")
        
        writer.add_scalar('Loss/train', loss, epoch)
        
        # 保存最佳模型（意图：基于验证损失保存最优检查点）
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
