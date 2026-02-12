"""
分词器模块

功能描述:
    定义中英文分词器，实现文本与词索引之间的转换。
    支持词表构建、文本编码、索引解码等功能。

核心组件:
    - BaseTokenizer: 分词器基类，定义通用接口
    - ChineseTokenizer: 中文分词器（字符级）
    - EnglishTokenizer: 英文分词器（词级）

特殊标记:
    <pad>: 填充标记，索引0
    <unk>: 未知词标记，索引1
    <sos>: 序列开始标记，索引2
    <eos>: 序列结束标记，索引3

作者: Red_Moon
创建日期: 2026-02
"""

import jieba
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer
from tqdm import tqdm


class BaseTokenizer:
    """
    分词器基类
    
    特殊标记:
        - <pad>: 填充标记，用于对齐不同长度的序列
        - <unk>: 未知词标记，处理词表外的词
        - <sos>: 序列开始标记，解码器的起始输入
        - <eos>: 序列结束标记，解码器的终止信号
    """
    
    pad_token = '<pad>'
    unk_token = '<unk>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    
    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        
        # 构建双向映射
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        
        # 特殊标记索引
        self.pad_token_index = self.word2index[self.pad_token]
        self.unk_token_index = self.word2index[self.unk_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]
    
    @classmethod
    def tokenize(cls, text):
        raise NotImplementedError
    
    def encode(self, text, add_sos_eos=False):
        """
        文本编码：将文本转换为词索引列表
        
        参数:
            text: 待编码文本
            add_sos_eos: 是否添加<sos>和<eos>标记
                        - True: 用于目标语言（解码器输入）
                        - False: 用于源语言（编码器输入）
        """
        tokens = self.tokenize(text)
        
        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        
        # 词表外的词使用<unk>索引
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]
    
    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        """
        构建词表
        
        词表格式:
            - 前4行固定为特殊标记: <pad>, <unk>, <sos>, <eos>
            - 其余词按出现顺序排列
        """
        vocab_set = set()
        
        for sentence in tqdm(sentences, desc="构建词表"):
            vocab_set.update(cls.tokenize(sentence))
        
        # 特殊标记在前
        vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token]
        vocab_list += [token for token in vocab_set if token.strip() != ""]
        
        print(f'词表大小: {len(vocab_list)}')
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))
    
    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)


class ChineseTokenizer(BaseTokenizer):
    """
    中文分词器
    
    分词策略:
        字符级分词（按字符切分）
    
    示例:
        输入: "我喜欢NLP"
        输出: ['我', '喜', '欢', 'N', 'L', 'P']
    """
    
    @classmethod
    def tokenize(cls, text):
        return list(text)


class EnglishTokenizer(BaseTokenizer):
    """
    英文分词器
    
    分词策略:
        使用NLTK的TreebankWordTokenizer
        支持处理缩写、标点等特殊情况
    
    示例:
        输入: "I don't like NLP!"
        输出: ['I', 'do', "n't", 'like', 'NLP', '!']
    """
    
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    
    @classmethod
    def tokenize(cls, text):
        return cls.tokenizer.tokenize(text)
    
    def decode(self, indexes):
        """索引解码为文本"""
        tokens = [self.index2word[index] for index in indexes]
        return self.detokenizer.detokenize(tokens)


if __name__ == '__main__':
    test_text = "Hello world!"
    tokens = EnglishTokenizer.tokenize(test_text)
    print(f"分词结果: {tokens}")
