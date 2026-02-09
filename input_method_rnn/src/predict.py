"""
模型预测模块
本模块负责加载训练好的模型，根据用户输入的前缀预测下一个词
"""
import torch
import jieba

import config
from model import InputMethodModel


def load_vocab():
    """
    加载词表
    
    从文件加载词表，并构建词到索引、索引到词的映射
    
    Returns:
        tuple: (word2index, index2word)
            - word2index (dict): 词到索引的映射
            - index2word (dict): 索引到词的映射
    """
    # 读取词表文件，每行一个词
    with open(config.MODELS_DIR / 'vocab.txt', 'r', encoding="utf-8") as f:
        vocab_list = [line.strip() for line in f]
    
    # 构建双向映射字典
    word2index = {word: index for index, word in enumerate(vocab_list)}
    index2word = {index: word for index, word in enumerate(vocab_list)}
    
    return word2index, index2word


def load_model(word2index):
    """
    加载模型
    
    加载训练好的模型权重，并设置为评估模式
    
    Args:
        word2index (dict): 词表映射，用于获取词表大小
    
    Returns:
        InputMethodModel: 加载好权重的模型实例
    """
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型实例
    vocab_size = len(word2index)
    model = InputMethodModel(vocab_size).to(device)
    
    # 加载预训练权重
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pth', map_location=device))
    
    # 设置为评估模式（禁用dropout等）
    model.eval()
    
    return model


def predict_next_word(prefix, model, word2index, index2word, top_k=5):
    """
    预测下一个词
    
    根据输入的前缀，预测概率最高的top_k个候选词
    
    Args:
        prefix (str): 输入前缀文本，如"我今天"
        model (InputMethodModel): 加载好的模型
        word2index (dict): 词到索引的映射
        index2word (dict): 索引到词的映射
        top_k (int): 返回候选词的数量，默认5
    
    Returns:
        list: 候选词列表，每个元素是(word, probability)元组
    """
    device = next(model.parameters()).device
    
    # Step 1: 分词 - 将输入文本切分为词序列
    tokens = jieba.lcut(prefix)
    
    # Step 2: 取最后SEQ_LEN个词作为输入
    # 如果输入长度超过SEQ_LEN，只保留最后SEQ_LEN个词
    # 如果输入长度不足SEQ_LEN，前面补<unk>（索引为0）
    if len(tokens) >= config.SEQ_LEN:
        tokens = tokens[-config.SEQ_LEN:]
    else:
        # 前面补0（<unk>的索引）
        tokens = ['<unk>'] * (config.SEQ_LEN - len(tokens)) + tokens
    
    # Step 3: 将词转换为索引
    input_ids = [word2index.get(token, 0) for token in tokens]
    
    # Step 4: 转换为张量并添加batch维度
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Step 5: 模型预测
    with torch.no_grad():
        output = model(input_tensor)  # 输出形状: [1, vocab_size]
        
        # 使用softmax获取概率分布
        probabilities = torch.softmax(output, dim=1)
        
        # 获取概率最高的top_k个词
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=1)
        
        # 转换为Python列表
        top_k_probs = top_k_probs.squeeze().cpu().numpy()
        top_k_indices = top_k_indices.squeeze().cpu().numpy()
    
    # Step 6: 将索引转换回词
    results = []
    for idx, prob in zip(top_k_indices, top_k_probs):
        word = index2word.get(int(idx), '<unk>')
        results.append((word, float(prob)))
    
    return results


if __name__ == '__main__':
    # ==================== 加载词表和模型 ====================
    print("正在加载词表和模型...")
    word2index, index2word = load_vocab()
    model = load_model(word2index)
    print("加载完成，可以开始预测")
    
    # ==================== 交互式预测 ====================
    while True:
        # 获取用户输入
        prefix = input("\n请输入前缀（输入'quit'退出）：")
        
        # 退出条件
        if prefix.lower() == 'quit':
            break
        
        # 预测下一个词
        predictions = predict_next_word(prefix, model, word2index, index2word, top_k=5)
        
        # 显示预测结果
        print(f"预测结果：")
        for i, (word, prob) in enumerate(predictions, 1):
            print(f"  {i}. {word} (概率: {prob:.4f})")
