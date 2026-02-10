"""
分词器模块

功能描述:
    本模块实现了基于jieba的中文分词器JiebaTokenizer，主要功能包括：
    1. 文本分词：使用jieba库进行中文分词
    2. 文本编码：将词序列转换为索引序列
    3. 索引解码：将索引序列转换回词序列
    4. 词表构建：从句子集合构建词表并保存到文件
    5. 词表加载：从文件加载预构建的词表

特殊标记:
    - <unk>: 未知词标记，用于处理词表外的词

作者: Red_Moon
创建日期: 2026-02
"""

import jieba
from tqdm import tqdm
import config


class JiebaTokenizer:
    """
    基于jieba的中文分词器

    功能描述:
        提供完整的中文文本处理功能，包括分词、编码、解码以及词表管理。
        使用jieba库进行精确模式分词，支持自定义词表的构建和加载。

    类属性:
        unk_token (str): 未知词标记，默认值为'<unk>'

    实例属性:
        vocab_list (list): 词表列表，按索引顺序排列
        vocab_size (int): 词表大小
        word2index (dict): 词到索引的映射字典
        index2word (dict): 索引到词的映射字典
        unk_token_index (int): 未知词标记的索引
    """

    unk_token = '<unk>'

    def __init__(self, vocab_list):
        """
        初始化分词器

        参数:
            vocab_list (list): 词表列表，第一个元素应为<unk>标记
        """
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}
        self.unk_token_index = self.word2index[self.unk_token]

    @staticmethod
    def tokenize(text):
        """
        文本分词

        功能描述:
            使用jieba库的精确模式对中文文本进行分词。
            jieba.lcut是jieba.cut的列表版本，直接返回列表而非生成器。

        参数:
            text (str): 待分词的中文文本

        返回:
            list: 分词结果列表，每个元素是一个词或字符

        分词模式:
            - 精确模式：试图将句子最精确地切开，适合文本分析
            - 全模式：把句子中所有可以成词的词语都扫描出来，速度快但不能解决歧义
            - 搜索引擎模式：在精确模式基础上，对长词再次切分，提高召回率
        """
        return jieba.lcut(text)

    def encode(self, text):
        """
        文本编码

        功能描述:
            将文本字符串编码为词索引序列。
            首先对文本进行分词，然后将每个词转换为其在词表中的索引。
            对于词表外的词(OOV)，使用<unk>标记的索引代替。

        参数:
            text (str): 待编码的文本字符串

        返回:
            list: 词索引列表，每个元素是整数索引

        OOV处理:
            如果词不在词表中，使用dict.get(key, default)方法返回unk_token_index
            这样可以避免KeyError异常
        """
        tokens = self.tokenize(text)
        return [self.word2index.get(token, self.unk_token_index) for token in tokens]

    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        """
        构建词表

        功能描述:
            从句子集合中构建词表并保存到文件。
            遍历所有句子，使用jieba分词收集所有唯一的词，
            然后将词表（包含<unk>标记）保存到指定路径。

        参数:
            sentences (list): 句子列表，每个句子是字符串
            vocab_path (str or Path): 词表文件保存路径

        返回:
            None

        词表构建流程:
            1. 遍历所有句子
            2. 对每个句子进行jieba分词
            3. 使用集合去重收集所有唯一的词
            4. 将<unk>标记添加到词表开头
            5. 保存词表到文件（每行一个词）

        文件格式:
            每行一个词，第一行是<unk>标记
            示例:
                <unk>
                今天
                天气
                ...
        """
        vocab_set = set()

        for sentence in tqdm(sentences, desc="构建词表"):
            vocab_set.update(jieba.lcut(sentence))

        vocab_list = [cls.unk_token] + list(vocab_set)
        print(f'词表大小: {len(vocab_list)}')

        with open(vocab_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    @classmethod
    def from_vocab(cls, vocab_path):
        """
        从文件加载词表

        功能描述:
            从词表文件加载词表并创建JiebaTokenizer实例。
            这是构建分词器实例的工厂方法。

        参数:
            vocab_path (str or Path): 词表文件路径

        返回:
            JiebaTokenizer: 使用加载的词表初始化的分词器实例

        文件格式要求:
            每行一个词，第一行应为<unk>标记

        异常处理:
            - 如果文件不存在，抛出FileNotFoundError
            - 如果文件编码错误，抛出UnicodeDecodeError
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]

        return cls(vocab_list)


if __name__ == '__main__':
    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')
    print(f'词表大小：{tokenizer.vocab_size}')
    print(f'特殊符号：{tokenizer.unk_token}')
    print(tokenizer.encode("今天天气不错"))
