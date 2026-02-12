"""
分词器模块

作者: Red_Moon
创建日期: 2026-02
"""

import jieba
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer
from tqdm import tqdm


class BaseTokenizer:
    """分词器基类"""
    
    # 特殊标记定义（意图：统一特殊标记，便于模型识别）
    pad_token = '<pad>'  # 填充标记
    unk_token = '<unk>'  # 未知词标记
    sos_token = '<sos>'  # 序列开始标记
    eos_token = '<eos>'  # 序列结束标记
    
    def __init__(self, vocab_list):
        """
        初始化分词器
        
        参数:
            vocab_list: 词表列表
        """
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        
        # 特殊标记索引（意图：快速访问常用标记）
        self.pad_token_index = self.word2index[self.pad_token]
        self.unk_token_index = self.word2index[self.unk_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]
    
    @classmethod
    def tokenize(cls, text):
        """分词方法（子类必须实现）"""
        raise NotImplementedError
    
    def encode(self, text, add_sos_eos=False):
        """
        文本编码
        
        参数:
            text: 待编码文本
            add_sos_eos: 是否添加<sos>和<eos>标记
        
        返回:
            list: 词索引列表
        """
        tokens = self.tokenize(text)
        if add_sos_eos:
            tokens = [self.sos_token] + tokens + [self.eos_token]
        
        # OOV处理：词表外词使用<unk>索引（警示：确保不会抛出KeyError）
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]
    
    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        """
        构建词表
        
        参数:
            sentences: 句子列表
            vocab_path: 词表保存路径
        """
        vocab_set = set()
        
        for sentence in tqdm(sentences, desc="构建词表"):
            vocab_set.update(cls.tokenize(sentence))
        
        # 特殊标记在前（意图：确保特殊标记索引固定且小，便于处理）
        vocab_list = [cls.pad_token, cls.unk_token, cls.sos_token, cls.eos_token]
        vocab_list += [token for token in vocab_set if token.strip() != ""]
        
        print(f'词表大小: {len(vocab_list)}')
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))
    
    @classmethod
    def from_vocab(cls, vocab_path):
        """
        从文件加载词表
        
        参数:
            vocab_path: 词表文件路径
        
        返回:
            BaseTokenizer: 分词器实例
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        
        return cls(vocab_list)


class ChineseTokenizer(BaseTokenizer):
    """中文分词器（意图：按字符级别处理中文，避免分词错误累积）"""
    
    @classmethod
    def tokenize(cls, text):
        """中文按字符分词"""
        return list(text)


class EnglishTokenizer(BaseTokenizer):
    """英文分词器（使用NLTK的TreebankWordTokenizer）"""
    
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()
    
    @classmethod
    def tokenize(cls, text):
        """英文分词"""
        return cls.tokenizer.tokenize(text)
    
    def decode(self, indexes):
        """
        索引解码为文本
        
        参数:
            indexes: 词索引列表
        
        返回:
            str: 解码后的文本
        """
        tokens = [self.index2word[index] for index in indexes]
        return self.detokenizer.detokenize(tokens)


if __name__ == '__main__':
    # 测试代码
    test_text = "Hello world!"
    tokens = EnglishTokenizer.tokenize(test_text)
    print(f"分词结果: {tokens}")
