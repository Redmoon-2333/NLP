
from torch.utils.tensorboard import SummaryWriter

# 创建写入器，指定日志保存目录
writer = SummaryWriter(log_dir="./logs")
for step in range(100):
    writer.add_scalar(tag="scaler/y=x", scalar_value=step, global_step=step)
    writer.add_scalar(tag="scaler/y=x^2", scalar_value=step ** 2, global_step=step)
writer.close()
