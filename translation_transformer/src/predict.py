"""
模型预测模块

功能描述:
    实现Transformer翻译模型的推理功能。
    包括自回归生成、批量预测、交互式翻译界面。

核心组件:
    - predict_batch: 批量预测函数（自回归生成）
    - predict: 单条文本预测
    - run_predict: 交互式预测界面

解码策略:
    - 贪心解码: 每步选择概率最高的词
    - 自回归生成: 每一步将之前所有已生成的词作为输入，预测下一个词
                  （区别于RNN：Transformer每次重新计算整个序列的注意力）

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
    
    功能:
        使用贪心解码策略，自回归地生成翻译结果。
        每次生成一个词，直到生成<eos>或达到最大长度。
    
    自回归生成流程:
        1. 编码源语言序列得到memory
        2. 初始化解码器输入为<sos>
        3. 循环生成：
           a. 解码得到输出概率
           b. 选择概率最高的词
           c. 添加到已生成序列
           d. 检查是否生成<eos>
        4. 截断<eos>之后的标记
    """
    model.eval()
    
    with torch.no_grad():
        # 编码阶段
        src_pad_mask = (inputs == model.zh_embedding.padding_idx)
        memory = model.encode(inputs, src_pad_mask)
        # memory.shape: [batch, src_len, d_model]

        batch_size = inputs.shape[0]
        device = inputs.device

        # 解码初始化
        # 初始化解码器输入为<sos>标记
        decoder_input = torch.full([batch_size, 1], en_tokenizer.sos_token_index, device=device)

        generated = []
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 自回归生成循环
        for i in range(config.SEQ_LEN):
            # 创建因果掩码
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
            
            # 解码
            decoder_output = model.decode(decoder_input, memory, tgt_mask, src_pad_mask)
            
            # 贪心解码：取最后一个位置的预测
            next_token_indexes = torch.argmax(decoder_output[:, -1, :], dim=-1, keepdim=True)
            
            generated.append(next_token_indexes)
            
            # 【自回归生成核心机制】
            # 将新生成的词拼接到已有序列末尾，形成新的完整输入序列
            # 
            # 自回归特性详解：
            # - 第1步：decoder_input = [<sos>]                    -> 预测 w1
            # - 第2步：decoder_input = [<sos>, w1]               -> 预测 w2
            # - 第3步：decoder_input = [<sos>, w1, w2]           -> 预测 w3
            # - 第i步：decoder_input = [<sos>, w1, w2, ..., w_i-1] -> 预测 w_i
            # 
            # 【与RNN的关键差异】
            # - RNN: decoder_input = next_token（仅传入上一个词）
            #        历史信息通过 hidden state 隐式传递
            # - Transformer: decoder_input = [sos, w1, w2, ..., w_i]（传入所有已生成的词）
            #                历史信息通过 Self-Attention 显式计算
            #                每步重新计算整个序列的注意力，而非增量计算
            decoder_input = torch.cat([decoder_input, next_token_indexes], dim=-1)
            
            # 检查终止条件
            is_finished |= (next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index)
            if is_finished.all():
                break
        
        # 处理预测结果
        generated_tensor = torch.cat(generated, dim=1)
        generated_list = generated_tensor.tolist()
        
        # 截断<eos>之后的标记
        for index, sentence in enumerate(generated_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]
        
        return generated_list


def predict(text, model, zh_tokenizer, en_tokenizer, device):
    """单条文本预测"""
    indexes = zh_tokenizer.encode(text)
    input_tensor = torch.tensor([indexes], dtype=torch.long).to(device)
    batch_result = predict_batch(model, input_tensor, en_tokenizer, device)
    return en_tokenizer.decode(batch_result[0])


def run_predict():
    """运行交互式预测界面"""
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
