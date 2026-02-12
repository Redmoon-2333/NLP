"""
模型评估模块

作者: Red_Moon
创建日期: 2026-02
"""

import torch
from nltk.translate.bleu_score import corpus_bleu

import config
from model import TranslationModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import ChineseTokenizer, EnglishTokenizer


def evaluate(model, test_dataloader, device, en_tokenizer):
    """
    评估模型在测试集上的性能
    
    参数:
        model: 已加载权重的模型
        test_dataloader: 测试数据加载器
        device: 计算设备
        en_tokenizer: 英文分词器
    
    返回:
        float: BLEU-4分数
    """
    predictions = []
    references = []
    
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        targets = targets.tolist()
        
        # 批量预测（意图：提高推理效率）
        batch_result = predict_batch(model, inputs, en_tokenizer, device)
        predictions.extend(batch_result)
        
        # 提取参考译文（去掉<sos>和<eos>之间的内容）
        for target in targets:
            # 找到<eos>位置，提取有效部分
            try:
                eos_pos = target.index(en_tokenizer.eos_token_index)
                ref = target[1:eos_pos]  # 去掉<sos>和<eos>
            except ValueError:
                ref = target[1:]  # 如果没有<eos>，只去掉<sos>
            references.append([ref])  # corpus_bleu需要嵌套列表格式
    
    # 计算BLEU-4（意图：使用NLTK的corpus_bleu计算语料级BLEU）
    return corpus_bleu(references, predictions)


def run_evaluate():
    """运行完整评估流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载词表
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    print("词表加载成功")
    
    # 初始化并加载模型
    model = TranslationModel(
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_index=zh_tokenizer.pad_token_index,
        en_padding_index=en_tokenizer.pad_token_index
    ).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))
    print("模型加载成功")
    
    # 加载测试集
    test_dataloader = get_dataloader(train=False)
    
    # 评估
    bleu = evaluate(model, test_dataloader, device, en_tokenizer)
    print(f"BLEU-4: {bleu:.4f}")


if __name__ == '__main__':
    run_evaluate()
