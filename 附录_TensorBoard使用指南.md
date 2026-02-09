# 附录：TensorBoard使用指南

## 第1章 概述

### 1.1 什么是TensorBoard？

**TensorBoard**是TensorFlow提供的一套可视化工具，用于理解、调试和优化机器学习模型。它能够将训练过程中的各种指标以直观的图表形式展示出来，帮助开发者监控模型训练状态、分析模型性能、调试问题。

**核心功能：**
- 📊 **可视化训练指标**：实时查看loss、accuracy等指标的变化趋势
- 🔍 **模型结构可视化**：查看网络结构和数据流向
- 📈 **参数分布可视化**：监控权重、梯度的统计分布
- 🖼️ **图像和文本可视化**：查看模型处理的数据和生成结果
- 🎯 **超参数调优**：对比不同超参数配置下的训练效果

**适用框架：**
虽然TensorBoard由TensorFlow开发，但它已经成为深度学习领域的通用工具，支持：
- PyTorch（通过torch.utils.tensorboard）
- Keras
- MXNet
- 其他框架

### 1.2 为什么需要TensorBoard？

**训练过程监控的挑战：**

在训练深度学习模型时，我们面临以下问题：

| 问题 | 说明 | TensorBoard的解决方案 |
|------|------|---------------------|
| **无法实时观察训练状态** | 只能看到终端输出的数字 | 实时图表展示训练曲线 |
| **难以对比实验** | 不同实验的结果难以比较 | 在同一图表中叠加多条曲线 |
| **调试困难** | 不知道模型哪里出了问题 | 可视化梯度、权重分布 |
| **结果不直观** | 数字堆砌，缺乏视觉冲击 | 图表、图像、文本展示 |

**使用前后对比：**

```
❌ 不使用TensorBoard：
Epoch 1/10, Loss: 2.3456, Acc: 0.5234
Epoch 2/10, Loss: 1.8923, Acc: 0.6123
Epoch 3/10, Loss: 1.4567, Acc: 0.6891
...

✅ 使用TensorBoard：
📊 平滑的Loss下降曲线
📈 清晰的Accuracy上升趋势
🔍 实时监控，发现异常立即处理
📋 多个实验对比一目了然
```

**典型应用场景：**
1. **训练监控**：实时查看loss是否收敛
2. **超参数调优**：对比不同学习率、batch size的效果
3. **模型诊断**：检查是否过拟合、梯度消失/爆炸
4. **结果展示**：向团队或论文展示训练效果

---

## 第2章 安装TensorBoard

### 2.1 安装方法

**方法一：通过pip安装（推荐）**

```bash
# 安装TensorBoard
pip install tensorboard

# 验证安装
tensorboard --version
```

**方法二：随PyTorch一起安装**

如果使用PyTorch，TensorBoard支持通常已包含：

```bash
# PyTorch环境中
pip install torch torchvision
# tensorboard已包含在依赖中

# 或明确安装
pip install tensorboard
```

**方法三：在虚拟环境中安装**

```bash
# 创建虚拟环境
conda create -n myenv python=3.8
conda activate myenv

# 安装TensorBoard
pip install tensorboard
```

### 2.2 验证安装

**测试TensorBoard是否正确安装：**

```python
# 创建一个简单的测试脚本 test_tensorboard.py
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 创建写入器
writer = SummaryWriter('runs/test')

# 写入一些测试数据
for i in range(100):
    writer.add_scalar('test/loss', np.sin(i/10), i)

writer.close()
print("测试数据已写入，请运行: tensorboard --logdir=runs")
```

**运行测试：**

```bash
# 1. 运行测试脚本
python test_tensorboard.py

# 2. 启动TensorBoard
tensorboard --logdir=runs

# 3. 在浏览器中打开
# 通常是 http://localhost:6006
```

如果能看到一条正弦曲线，说明安装成功！

### 2.3 常见安装问题

**问题1：端口被占用**

```bash
# 错误信息
TensorBoard attempted to bind to port 6006, but it was already in use

# 解决方案：指定其他端口
tensorboard --logdir=runs --port=6007
```

**问题2：找不到tensorboard命令**

```bash
# 确认安装路径
pip show tensorboard

# 将tensorboard添加到PATH
# Windows:
set PATH=%PATH%;C:\Users\YourName\AppData\Local\Programs\Python\Python38\Scripts

# Linux/Mac:
export PATH=$PATH:~/.local/bin
```

**问题3：版本冲突**

```bash
# 卸载旧版本
pip uninstall tensorboard

# 重新安装
pip install tensorboard
```

---

## 第3章 基础使用

### 3.1 概述

使用TensorBoard的基本流程包括三个步骤：

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  1. 记录数据     │  →   │  2. 启动服务     │  →   │  3. 浏览器查看   │
│  (训练代码中)    │      │  (命令行)        │      │  (Web界面)       │
└─────────────────┘      └─────────────────┘      └─────────────────┘
  SummaryWriter         tensorboard --logdir      http://localhost:6006
```

### 3.2 创建SummaryWriter

**SummaryWriter**是TensorBoard的核心类，负责将训练数据写入日志文件。

**基本用法：**

```python
from torch.utils.tensorboard import SummaryWriter

# 创建写入器（指定日志目录）
writer = SummaryWriter('runs/experiment_1')

# ... 训练代码 ...

# 关闭写入器
writer.close()
```

**参数说明：**

```python
writer = SummaryWriter(
    log_dir='runs/my_experiment',  # 日志保存目录
    comment='learning_rate_0.001', # 实验备注（会添加到目录名）
    flush_secs=10                  # 多少秒刷新一次到磁盘
)
```

**目录结构示例：**

```
项目根目录/
└── runs/                          # 默认日志根目录
    ├── experiment_1/              # 实验1的日志
    │   └── events.out.tfevents.*  # TensorBoard事件文件
    ├── experiment_2/              # 实验2的日志
    └── Jan01_12-00-00_hostname/   # 自动命名的实验
```

**最佳实践：**

```python
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 使用时间戳命名，避免覆盖
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = os.path.join('runs', f'exp_{timestamp}')
writer = SummaryWriter(log_dir)

print(f"TensorBoard日志保存在: {log_dir}")
print(f"启动命令: tensorboard --logdir=runs")
```

### 3.3 记录标量（Scalar）

**标量**是最常用的记录类型，用于记录单个数值（如loss、accuracy）随时间的变化。

**基本用法：**

```python
# add_scalar(tag, scalar_value, global_step)
# tag: 标签名称
# scalar_value: 要记录的数值
# global_step: 横轴坐标（通常是迭代次数或epoch）

writer.add_scalar('Loss/train', loss.item(), epoch)
writer.add_scalar('Accuracy/train', acc, epoch)
```

**完整训练示例：**

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 创建写入器
writer = SummaryWriter('runs/mnist_experiment')

# 模拟训练过程
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    # 模拟前向传播
    inputs = torch.randn(32, 10)
    labels = torch.randn(32, 1)
    
    # 计算损失
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

writer.close()
```

**组织标量的命名规范：**

使用`/`分隔可以创建层次结构，TensorBoard会自动分组：

```python
# 训练集和测试集分开
writer.add_scalar('Loss/train', train_loss, step)
writer.add_scalar('Loss/test', test_loss, step)

# 不同指标分组
writer.add_scalar('Metrics/accuracy', acc, step)
writer.add_scalar('Metrics/precision', prec, step)
writer.add_scalar('Metrics/recall', recall, step)

# 多任务学习
writer.add_scalar('Task1/loss', task1_loss, step)
writer.add_scalar('Task2/loss', task2_loss, step)
```

**在TensorBoard中的展示：**

```
📊 SCALARS标签页
├─ Loss
│  ├─ train  📈 (训练loss曲线)
│  └─ test   📈 (测试loss曲线)
└─ Metrics
   ├─ accuracy  📈
   ├─ precision 📈
   └─ recall    📈
```

### 3.4 自动TensorBoard服务

**手动启动TensorBoard：**

```bash
# 基本命令
tensorboard --logdir=runs

# 指定端口
tensorboard --logdir=runs --port=6007

# 指定主机（允许远程访问）
tensorboard --logdir=runs --host=0.0.0.0

# 后台运行
nohup tensorboard --logdir=runs &
```

**在Jupyter Notebook中使用：**

```python
# 方法1：使用魔法命令（推荐）
%load_ext tensorboard
%tensorboard --logdir runs

# 方法2：使用notebook模块
from tensorboard import notebook
notebook.start("--logdir runs")
```

**在Python代码中自动启动：**

```python
import subprocess
import webbrowser
import time

def start_tensorboard(logdir='runs', port=6006):
    """自动启动TensorBoard并打开浏览器"""
    # 启动TensorBoard进程
    process = subprocess.Popen(
        ['tensorboard', '--logdir', logdir, '--port', str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待服务启动
    time.sleep(3)
    
    # 自动打开浏览器
    url = f'http://localhost:{port}'
    webbrowser.open(url)
    
    print(f"TensorBoard已启动: {url}")
    return process

# 使用
tb_process = start_tensorboard()

# 训练结束后关闭
# tb_process.terminate()
```

---

## 第4章 参考资料

### 4.1 官方文档

- **TensorBoard官方文档**: https://www.tensorflow.org/tensorboard
- **PyTorch TensorBoard教程**: https://pytorch.org/docs/stable/tensorboard.html
- **GitHub仓库**: https://github.com/tensorflow/tensorboard

### 4.2 常用资源

**教程和示例：**
- TensorFlow官方教程：完整的使用指南和最佳实践
- PyTorch官方示例：PyTorch集成TensorBoard的示例代码
- Keras文档：Keras中使用TensorBoard回调

**社区资源：**
- Stack Overflow：常见问题解答
- Reddit r/MachineLearning：经验分享
- 知乎专栏：中文教程和案例

### 4.3 实用技巧

**技巧1：实验管理**

```python
# 使用配置字典组织实验
config = {
    'lr': 0.001,
    'batch_size': 64,
    'optimizer': 'Adam'
}

# 将配置编码到目录名
from urllib.parse import urlencode
config_str = urlencode(config)
log_dir = f'runs/exp_{config_str}'
writer = SummaryWriter(log_dir)
```

**技巧2：清理旧日志**

```python
import shutil

# 删除旧的实验日志
def clean_old_runs(keep_recent=5):
    runs_dir = 'runs'
    subdirs = sorted(os.listdir(runs_dir))
    
    if len(subdirs) > keep_recent:
        for old_dir in subdirs[:-keep_recent]:
            shutil.rmtree(os.path.join(runs_dir, old_dir))
            print(f"已删除旧日志: {old_dir}")
```

**技巧3：多GPU训练中只在主进程记录**

```python
import torch.distributed as dist

# 只在rank 0进程记录
if not dist.is_initialized() or dist.get_rank() == 0:
    writer = SummaryWriter('runs/experiment')
else:
    writer = None

# 训练中
if writer is not None:
    writer.add_scalar('Loss/train', loss.item(), step)
```

**技巧4：使用上下文管理器**

```python
from contextlib import contextmanager

@contextmanager
def create_summary_writer(log_dir):
    writer = SummaryWriter(log_dir)
    try:
        yield writer
    finally:
        writer.close()
        print(f"TensorBoard日志已保存到: {log_dir}")

# 使用
with create_summary_writer('runs/exp') as writer:
    for epoch in range(100):
        # 训练代码
        writer.add_scalar('Loss', loss, epoch)
```

### 4.4 命令速查表

**启动命令：**

```bash
# 基本启动
tensorboard --logdir=runs

# 指定端口
tensorboard --logdir=runs --port=6007

# 允许远程访问
tensorboard --logdir=runs --host=0.0.0.0

# 后台运行
tensorboard --logdir=runs &

# 查看版本
tensorboard --version

# 查看帮助
tensorboard --help
```

**常用Python API：**

```python
from torch.utils.tensorboard import SummaryWriter

# 创建写入器
writer = SummaryWriter('runs/exp')

# 记录标量
writer.add_scalar('Loss', loss, step)

# 记录多个标量
writer.add_scalars('Losses', {'train': train_loss, 'test': test_loss}, step)

# 记录直方图
writer.add_histogram('weights', model.fc.weight, step)

# 记录图像
writer.add_image('Image', img_tensor, step)

# 记录图
writer.add_graph(model, input_tensor)

# 记录文本
writer.add_text('Config', 'Learning rate: 0.001', step)

# 关闭写入器
writer.close()
```

---

## 附录：完整项目示例

### 示例：使用TensorBoard监控RNN训练

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

def train_with_tensorboard():
    # 1. 创建TensorBoard写入器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('runs', f'rnn_experiment_{timestamp}')
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard日志目录: {log_dir}")
    print(f"启动命令: tensorboard --logdir=runs")
    
    # 2. 模型和训练配置
    model = SimpleRNN(input_size=10, hidden_size=20, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. 记录超参数
    hparams = {
        'lr': 0.001,
        'batch_size': 32,
        'hidden_size': 20,
        'optimizer': 'Adam'
    }
    writer.add_text('Hyperparameters', str(hparams), 0)
    
    # 4. 训练循环
    for epoch in range(100):
        # 模拟训练数据
        inputs = torch.randn(32, 5, 10)  # (batch, seq_len, input_size)
        labels = torch.randint(0, 2, (32,))  # (batch,)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 5. 记录到TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('Accuracy/train', accuracy.item(), epoch)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # 每10个epoch记录权重分布
        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'Parameters/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss={loss.item():.4f}, Acc={accuracy.item():.4f}')
    
    # 6. 关闭写入器
    writer.close()
    print("训练完成！请在浏览器中查看TensorBoard")

if __name__ == '__main__':
    train_with_tensorboard()
```

**运行步骤：**

```bash
# 1. 运行训练脚本
python train.py

# 2. 启动TensorBoard（在另一个终端）
tensorboard --logdir=runs

# 3. 在浏览器中打开
# http://localhost:6006
```

**预期结果：**
- 📉 Loss曲线平滑下降
- 📈 Accuracy曲线稳步上升
- 📊 权重和梯度的分布直方图
- 📝 超参数配置记录

---

**恭喜！** 你已经掌握了TensorBoard的基础使用。继续探索更多高级功能，让模型训练过程更加透明和可控！

**下一步建议：**
1. 尝试在自己的项目中集成TensorBoard
2. 探索图像、音频等多媒体数据的可视化
3. 学习使用TensorBoard进行超参数调优
4. 研究如何在生产环境中使用TensorBoard监控模型

**Happy Visualizing! 📊**
