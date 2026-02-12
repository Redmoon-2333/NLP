"""
模型预测模块

作者: Red_Moon
创建日期: 2026-02
"""

import torch
import config
from model import TranslationModel
from tokenizer import ChineseTokenizer, EnglishTokenizer


def predict_batch(model, inputs, en_tokenizer, device):
    """
    批量预测（自回归生成）
    
    参数:
        model: 已加载权重的模型
        inputs: [batch_size, seq_len] 源语言输入
        en_tokenizer: 英文分词器
        device: 计算设备
    
    返回:
        list: 预测的词索引列表
    """
    model.eval()
    
    with torch.no_grad():
        # 编码阶段（意图：将源语言编码为上下文向量）
        encoder_outputs,context_vector = model.encoder(inputs)  # [batch, hidden_size]
        
        batch_size = inputs.shape[0]
        hidden = context_vector.unsqueeze(0)  # [1, batch, hidden_size]
        
        # 初始化解码器输入为<sos>标记（意图：自回归生成的起始信号）
        decoder_input = torch.full([batch_size, 1], en_tokenizer.sos_token_index, device=device)
        
        generated = []  # 存储生成的词索引
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)  # 记录已完成序列
        
        # 自回归生成（警示：最大长度限制防止无限生成）
        for i in range(config.SEQ_LEN):
            decoder_output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
            # decoder_output: [batch, 1, vocab_size]
            
            # 贪心解码：选择概率最高的词（意图：简单高效的解码策略）
            next_token_indexes = torch.argmax(decoder_output, dim=-1)  # [batch, 1]
            generated.append(next_token_indexes)
            
            # 更新输入（意图：自回归特性，当前输出作为下一步输入）
            decoder_input = next_token_indexes
            
            # 检查是否生成<eos>（意图：提前终止已完成序列）
            is_finished |= (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
            if is_finished.all():
                break
        
        # 处理预测结果
        generated_tensor = torch.cat(generated, dim=1)  # [batch, seq_len]
        generated_list = generated_tensor.tolist()
        
        # 截断<eos>之后的标记（意图：清理输出，只保留有效部分）
        for index, sentence in enumerate(generated_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]
        
        return generated_list


def predict(text, model, zh_tokenizer, en_tokenizer, device):
    """
    单条文本预测
    
    参数:
        text: 待翻译的中文文本
        model: 已加载权重的模型
        zh_tokenizer: 中文分词器
        en_tokenizer: 英文分词器
        device: 计算设备
    
    返回:
        str: 翻译后的英文文本
    """
    indexes = zh_tokenizer.encode(text)
    input_tensor = torch.tensor([indexes], dtype=torch.long).to(device)
    
    batch_result = predict_batch(model, input_tensor, en_tokenizer, device)
    return en_tokenizer.decode(batch_result[0])


def run_predict():
    """运行交互式预测界面"""
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
    
    print("\n" + "=" * 40)
    print("欢迎使用翻译模型(输入q或者quit退出)")
    print("=" * 40)
    
    while True:
        user_input = input("中文： ")
        
        if user_input in ['q', 'quit']:
            print("欢迎下次再来")
            break
        
        if user_input.strip() == '':
            print("请输入内容")
            continue
        
        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print(f"翻译结果: {result}")
        print("-" * 40)


if __name__ == '__main__':
    run_predict()
