"""
模型评估模块

功能描述:
    实现Transformer翻译模型的评估功能。
    使用BLEU-4指标评估翻译质量。

核心组件:
    - evaluate: 评估函数
    - run_evaluate: 完整评估流程

评估指标:
    BLEU (Bilingual Evaluation Understudy):
        - 衡量机器翻译与参考译文的相似度
        - BLEU-4: 考虑1-4元语法
        - 取值范围: [0, 1]，越高越好

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
        
        # 批量预测
        batch_result = predict_batch(model, inputs, en_tokenizer, device)
        predictions.extend(batch_result)
        
        # 提取参考译文
        for target in targets:
            try:
                eos_pos = target.index(en_tokenizer.eos_token_index)
                ref = target[1:eos_pos]  # 去掉<sos>和<eos>
            except ValueError:
                ref = target[1:]
            
            # corpus_bleu需要嵌套列表格式
            references.append([ref])
    
    # 计算BLEU-4
    return corpus_bleu(references, predictions)


def run_evaluate():
    """运行完整评估流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    zh_tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    print("词表加载成功")
    
    model = TranslationModel(
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_index=zh_tokenizer.pad_token_index,
        en_padding_index=en_tokenizer.pad_token_index
    ).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth'))
    print("模型加载成功")
    
    test_dataloader = get_dataloader(train=False)
    
    bleu = evaluate(model, test_dataloader, device, en_tokenizer)
    print(f"BLEU-4: {bleu:.4f}")


if __name__ == '__main__':
    run_evaluate()
