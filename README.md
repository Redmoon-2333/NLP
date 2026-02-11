# NLP 自然语言处理学习项目

一个系统化的自然语言处理学习仓库，涵盖从基础概念到实践应用的完整知识体系。

**作者**: Red_Moon  
**开发时间**: 2026年2月

---

## 📚 项目简介

本项目旨在帮助初学者系统学习NLP（自然语言处理）技术，从基础概念出发，逐步深入到文本表示、序列模型、深度学习等核心内容。所有内容均配有详细的理论讲解和实践代码。

---

## 📁 项目结构

```
NLP/
├── 01_NLP导论.md              # 第1章：NLP基础概念与任务
├── 02_文本表示.md              # 第2章：文本表示方法详解
├── 03_RNN.md                   # 第3章：RNN循环神经网络详解
├── 03_LSTM.md                  # 第3章：LSTM长短期记忆网络详解
├── 03_GRU.md                   # 第3章：GRU门控循环单元详解
├── 04_Seq2Seq.md               # 第4章：Seq2Seq序列到序列模型
├── 附录_TensorBoard使用指南.md  # TensorBoard可视化工具指南
├── 附录_BLEU使用指南.md         # BLEU翻译质量评估指南
├── text_rep/                   # 文本表示实践代码
│   ├── 01_tokenize_jieba.ipynb    # 中文分词实践
│   ├── 02_word_representation.ipynb # 词向量与Word2Vec实践
│   └── data/                      # 数据集
├── input_method_rnn/           # 智能输入法RNN项目
│   ├── src/                       # 源代码
│   ├── data/                      # 数据目录
│   ├── models/                    # 模型保存目录
│   └── logs/                      # 训练日志
├── review_analyze_lstm/        # 评论情感分析LSTM项目
│   ├── src/                       # 源代码
│   ├── data/                      # 数据目录
│   ├── models/                    # 模型保存目录
│   └── logs/                      # 训练日志
├── translation_seq2seq/        # 中英机器翻译Seq2Seq项目
│   ├── src/                       # 源代码
│   │   ├── config.py                 # 配置文件
│   │   ├── process.py                # 数据预处理
│   │   ├── dataset.py                # Dataset和DataLoader
│   │   ├── model.py                  # Seq2Seq模型定义
│   │   ├── train.py                  # 训练流程
│   │   ├── evaluate.py               # BLEU评估
│   │   ├── predict.py                # 预测接口
│   │   └── tokenizer.py              # 中英文分词器
│   ├── data/                      # 数据目录
│   ├── models/                    # 模型保存目录
│   └── logs/                      # TensorBoard训练日志
├── 2.资料/                     # 学习资料
│   ├── 2.数据集/               # 各类数据集
│   └── 3.预训练模型/           # 预训练模型
└── README.md                   # 项目说明
```

---

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

#### 3.1 RNN（循环神经网络）
- RNN基础结构与原理
- 多层RNN与双向RNN
- 输入输出模式详解
- API使用与实战案例（智能输入法）
- RNN的局限性（梯度消失、长期依赖）

#### 3.2 LSTM（长短期记忆网络）
- LSTM的核心结构（遗忘门、输入门、输出门）
- 细胞状态与门控机制
- 双向LSTM与多层LSTM
- API使用与实战案例（评论情感分析）

#### 3.3 GRU（门控循环单元）
- GRU的简化结构
- 更新门与重置门
- 双向GRU与多层GRU
- LSTM vs GRU对比

### 第4章：Seq2Seq序列到序列模型
- **Seq2Seq架构**：编码器-解码器结构
- **编码器**：将输入序列压缩为上下文向量
- **解码器**：自回归生成输出序列
- **Teacher Forcing**：训练技巧与Exposure Bias问题
- **推理策略**：贪心搜索、束搜索(Beam Search)
- **实战项目**：中英机器翻译系统
  - 数据预处理与词表构建
  - GRU编码器-解码器实现
  - BLEU-4质量评估
  - 交互式翻译界面

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- Jupyter Notebook

### 安装依赖
```bash
pip install jieba pandas gensim transformers torch tqdm scikit-learn nltk sacrebleu
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

#### 4. 评论情感分析LSTM项目
```bash
# 进入项目目录
cd review_analyze_lstm

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
========================================
欢迎使用情感分析模型(输入q或者quit退出)
========================================
> 这款手机质量太差了，完全不值这个价
预测结果: 0.12
负向评论,置信度:0.88
----------------------------------------
> 非常满意，物流很快，商品质量很好
预测结果: 0.91
正向评论,置信度:0.91
```

#### 5. 中英机器翻译Seq2Seq项目
```bash
# 进入项目目录
cd translation_seq2seq

# 1. 数据预处理
python src/process.py

# 2. 训练模型
python src/train.py

# 3. 评估模型（BLEU-4）
python src/evaluate.py

# 4. 交互式翻译
python src/predict.py
```

翻译效果示例：
```
========================================
欢迎使用翻译模型(输入q或者quit退出)
========================================
中文： 你好世界
翻译结果: Hello world
----------------------------------------
中文： 我喜欢自然语言处理
翻译结果: I like natural language processing
----------------------------------------
```

---

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
├─ 阅读 03_RNN.md
│   ├─ 理解RNN的循环结构和局限性
│   └─ 完成智能输入法项目
├─ 阅读 03_LSTM.md
│   ├─ 学习LSTM的门控机制
│   └─ 完成评论情感分析项目
├─ 阅读 03_GRU.md
│   └─ 了解GRU的简化设计
└─ 掌握多层RNN和双向RNN的使用

第4阶段：Seq2Seq与机器翻译
├─ 阅读 04_Seq2Seq.md
│   ├─ 理解编码器-解码器架构
│   ├─ 学习Teacher Forcing和推理策略
│   └─ 完成中英翻译项目
├─ 阅读 附录_TensorBoard使用指南.md
│   └─ 掌握训练过程可视化
└─ 阅读 附录_BLEU使用指南.md
    └─ 掌握翻译质量评估方法

第5阶段：深入探索
├─ 学习Attention机制
├─ 了解Transformer架构
├─ 探索预训练语言模型（BERT、GPT）
└─ 尝试应用到实际任务中
```

---

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
| **Seq2Seq** | 序列到序列模型，编码器-解码器架构 |
| **Teacher Forcing** | 训练时使用真实标签作为解码器输入 |
| **Beam Search** | 束搜索，保留多个候选序列的解码策略 |
| **BLEU** | 机器翻译质量评估指标 |
| **注意力机制** | 动态关注输入不同部分的技术 |

---

## 📊 数据集说明

| 数据集 | 描述 | 用途 |
|--------|------|------|
| `online_shopping_10_cats.csv` | 10类商品在线评论数据 | 训练词向量、情感分析 |
| `user_dict.txt` | 自定义词典 | 提升分词准确率 |
| `cmn.txt` | 中英短句对照 | 机器翻译练习 |
| `synthesized_.jsonl` | 对话数据 | 对话系统、智能输入法训练 |

---

## 📚 推荐资源

### 预训练模型
- **bert-base-chinese**: 中文BERT基础模型
- **sgns.weibo.word**: 微博词向量（SGNS算法，300维）

### 学习资料
- 理论文档：`01_NLP导论.md`、`02_文本表示.md`、`03_RNN.md`、`03_LSTM.md`、`03_GRU.md`、`04_Seq2Seq.md`
- 实践代码：`text_rep/*.ipynb`、`input_method_rnn/src/*.py`、`review_analyze_lstm/src/*.py`、`translation_seq2seq/src/*.py`
- 工具指南：`附录_TensorBoard使用指南.md`、`附录_BLEU使用指南.md`

---

## 📝 更新日志

- **2026-02**: 项目初始化，添加NLP导论和文本表示理论文档
- **2026-02**: 添加Jieba分词和Word2Vec实践代码
- **2026-02**: 添加RNN、LSTM、GRU理论文档
- **2026-02**: 添加智能输入法RNN项目完整代码
- **2026-02**: 添加评论情感分析LSTM项目完整代码
- **2026-02**: 添加Seq2Seq理论文档和中英翻译项目
- **2026-02**: 添加TensorBoard使用指南附录
- **2026-02**: 添加BLEU使用指南附录

---

## 🤝 贡献

欢迎提交Issue和Pull Request，一起完善这个NLP学习项目！

---

**Happy Learning NLP! 🎉**
