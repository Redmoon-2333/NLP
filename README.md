# NLP 自然语言处理学习项目

一个系统化的自然语言处理学习仓库，涵盖从基础概念到实践应用的完整知识体系。

## 📚 项目简介

本项目旨在帮助初学者系统学习NLP（自然语言处理）技术，从基础概念出发，逐步深入到文本表示、序列模型、深度学习等核心内容。所有内容均配有详细的理论讲解和实践代码。

## 📁 项目结构

```
NLP/
├── 01_NLP导论.md              # 第1章：NLP基础概念与任务
├── 02_文本表示.md              # 第2章：文本表示方法详解
├── 03_传统序列模型.md           # 第3章：RNN、LSTM、GRU详解
├── text_rep/                   # 实践代码
│   ├── 01_tokenize_jieba.ipynb    # 中文分词实践
│   ├── 02_word_representation.ipynb # 词向量与Word2Vec实践
│   └── data/                      # 数据集
│       ├── online_shopping_10_cats.csv  # 在线购物评论数据集
│       ├── user_dict.txt                # 自定义词典
│       └── word2vec.txt                 # 训练好的词向量
├── input_method_rnn/           # 智能输入法RNN项目
│   ├── src/                       # 源代码
│   │   ├── config.py                 # 配置文件
│   │   ├── process.py                # 数据预处理
│   │   ├── dataset.py                # 数据集类
│   │   ├── model.py                  # RNN模型定义
│   │   ├── train.py                  # 训练脚本
│   │   ├── evaluate.py               # 评估脚本
│   │   └── predict.py                # 预测脚本
│   ├── data/                      # 数据目录
│   ├── models/                    # 模型保存目录
│   └── logs/                      # 训练日志
├── 2.资料/                     # 学习资料
│   ├── 2.数据集/               # 各类数据集
│   │   ├── 1.评论数据集/
│   │   ├── 2.对话数据集/
│   │   └── 3.中英短句数据集/
│   └── 3.预训练模型/           # 预训练模型
│       └── bert-base-chinese/
└── README.md                   # 项目说明
```

## 🎯 学习内容

### 第1章：NLP导论
- **NLP定义与目标**：理解什么是自然语言处理及其核心挑战
- **常见任务**：
  - 基础任务：分词、词性标注、命名实体识别(NER)
  - 理解类任务：情感分析、文本分类、问答系统
  - 生成类任务：机器翻译、文本摘要、对话系统
- **技术演进历史**：从规则方法→统计方法→深度学习→大模型时代

### 第2章：文本表示
- **分词(Tokenization)**：
  - 中文分词方法（规则、统计、深度学习）
  - 英文分词与子词分词（BPE、WordPiece、SentencePiece）
  - Jieba分词工具使用
- **词表示方法**：
  - 离散表示：One-Hot编码、词袋模型(BoW)、N-gram
  - 分布式表示：Word2Vec(CBOW/Skip-gram)、GloVe、FastText
  - 上下文词向量：ELMo、BERT、GPT

### 第3章：传统序列模型
- **RNN（循环神经网络）**：
  - RNN基础结构与原理
  - 多层RNN与双向RNN
  - 输入输出模式详解
  - API使用与实战案例（智能输入法）
  - RNN的局限性（梯度消失、长期依赖）
- **LSTM（长短期记忆网络）**：
  - LSTM的核心结构（遗忘门、输入门、输出门）
  - 细胞状态与门控机制
  - 双向LSTM与多层LSTM
- **GRU（门控循环单元）**：
  - GRU的简化结构
  - 更新门与重置门
  - LSTM vs GRU对比

## 🚀 快速开始

### 环境要求
- Python 3.7+
- Jupyter Notebook

### 安装依赖
```bash
pip install jieba pandas gensim transformers torch
```

### 运行示例

#### 1. 中文分词
```python
import jieba

# 精确模式
text = "小明毕业于北京大学计算机系"
words = jieba.lcut(text)
print(words)  # ['小明', '毕业', '于', '北京大学', '计算机系']

# 加载自定义词典
jieba.load_userdict("./data/user_dict.txt")
```

#### 2. 使用预训练词向量
```python
from gensim.models import KeyedVectors

# 加载预训练词向量
model = KeyedVectors.load_word2vec_format('./data/sgns.weibo.word')

# 计算词语相似度
similarity = model.similarity("公交", "地铁")

# 类比推理
result = model.most_similar(positive=['男人','女孩'], negative=['男孩'])
```

#### 3. 智能输入法RNN项目
```bash
# 进入项目目录
cd input_method_rnn

# 1. 数据预处理
python src/process.py

# 2. 训练模型
python src/train.py

# 3. 评估模型
python src/evaluate.py

# 4. 交互式预测
python src/predict.py
```

预测效果示例：
```
请输入前缀（输入'quit'退出）：我今天
预测结果：
  1. 去 (概率: 0.3124)
  2. 要 (概率: 0.2856)
  3. 想 (概率: 0.1987)
  4. 感觉 (概率: 0.1023)
  5. 已经 (概率: 0.0562)
```

## 📖 学习路径建议

```
第1阶段：基础概念
├─ 阅读 01_NLP导论.md
├─ 了解NLP的定义、任务和发展历史
└─ 理解NLP的挑战性（歧义性、上下文依赖等）

第2阶段：文本表示基础
├─ 阅读 02_文本表示.md
├─ 学习分词原理和方法
├─ 理解Token、词表、嵌入层等核心概念
└─ 掌握Word2Vec词向量训练和PyTorch Embedding使用

第3阶段：序列模型
├─ 阅读 03_传统序列模型.md
├─ 理解RNN的循环结构和局限性
├─ 学习LSTM的门控机制
├─ 了解GRU的简化设计
└─ 掌握多层RNN和双向RNN的使用

第4阶段：动手实践
├─ 运行 text_rep/*.ipynb
│   └─ 掌握jieba分词和词向量训练
├─ 完成 input_method_rnn 项目
│   └─ 实现基于RNN的智能输入法
└─ 尝试修改参数，观察效果变化

第5阶段：深入探索
├─ 学习Attention机制
├─ 了解Transformer架构
├─ 探索预训练语言模型（BERT、GPT）
└─ 尝试应用到实际任务中
```

## 🔧 核心概念速查

| 概念 | 说明 |
|------|------|
| **Token** | 文本的最小单位（词、子词或字符） |
| **词表(Vocabulary)** | Token到ID的映射表 |
| **嵌入层(Embedding)** | 将Token ID转换为向量的查找表 |
| **词向量** | 词的数值向量表示，捕捉语义信息 |
| **分词** | 将文本切分成Token的过程 |
| **OOV** | Out-of-Vocabulary，未登录词 |
| **CBOW** | 用上下文预测中心词 |
| **Skip-gram** | 用中心词预测上下文 |
| **RNN** | 循环神经网络，处理序列数据 |
| **LSTM** | 长短期记忆网络，解决RNN长期依赖问题 |
| **GRU** | 门控循环单元，LSTM的简化版本 |
| **双向RNN** | 同时考虑过去和未来上下文的RNN |
| **多层RNN** | 堆叠多个RNN层，学习层次化特征 |

## 📊 数据集说明

| 数据集 | 描述 | 用途 |
|--------|------|------|
| `online_shopping_10_cats.csv` | 10类商品在线评论数据 | 训练词向量、情感分析 |
| `user_dict.txt` | 自定义词典 | 提升分词准确率 |
| `cmn.txt` | 中英短句对照 | 机器翻译练习 |
| `synthesized_.jsonl` | 对话数据 | 对话系统、智能输入法训练 |

## 📚 推荐资源

### 预训练模型
- **bert-base-chinese**: 中文BERT基础模型
- **sgns.weibo.word**: 微博词向量（SGNS算法，300维）

### 学习资料
- 理论文档：`01_NLP导论.md`、`02_文本表示.md`、`03_传统序列模型.md`
- 实践代码：`text_rep/*.ipynb`、`input_method_rnn/src/*.py`

## 📝 更新日志

- **2024-02**: 项目初始化，添加NLP导论和文本表示理论文档
- **2024-02**: 添加Jieba分词和Word2Vec实践代码
- **2024-02**: 添加传统序列模型文档（RNN、LSTM、GRU）
- **2024-02**: 添加智能输入法RNN项目完整代码

---

## 🤝 贡献

欢迎提交Issue和Pull Request，一起完善这个NLP学习项目！

---

**Happy Learning NLP! 🎉**
