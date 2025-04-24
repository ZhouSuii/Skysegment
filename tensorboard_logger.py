# 创建新文件 utils/tensorboard_logger.py
import torch
import time
import os
import psutil
from torch.utils.tensorboard import SummaryWriter


def log_system_info(self, step=None):
    """记录系统信息"""
    if step is None:
        step = self.step_count

    # CPU利用率
    cpu_percent = psutil.cpu_percent()
    self.log_scalar("system/cpu_percent", cpu_percent, step)

    # 内存利用率
    memory = psutil.virtual_memory()
    self.log_scalar("system/memory_percent", memory.percent, step)

    # GPU利用率(如果可用)
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated(0) / 1e9  # GB
        gpu_reserved = torch.cuda.memory_reserved(0) / 1e9  # GB

        self.log_scalar("system/gpu_allocated_gb", gpu_allocated, step)
        self.log_scalar("system/gpu_reserved_gb", gpu_reserved, step)


class TensorboardLogger:
    def __init__(self, config=None):
        """初始化 TensorBoard 日志记录器"""
        # 如果没有配置，使用默认值
        if config is None:
            config = {}

        # 获取记录频率配置
        self.log_freq = config.get("log_freq", 1)  # 每隔多少个 episode 记录一次
        self.histogram_freq = config.get("histogram_freq", 10)  # 每隔多少个 episode 记录一次直方图

        # 创建日志目录
        self.log_dir = config.get("log_dir", "runs")
        self.experiment_name = config.get("experiment_name", "graph_partition_experiment")
        timestamp = str(int(time.time()))
        self.log_path = f"{self.log_dir}/{self.experiment_name}_{timestamp}"

        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化 SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_path)

        # 记录步数计数器
        self.episode_count = 0
        self.step_count = 0

    def log_scalar(self, tag, value, step=None):
        """记录标量值"""
        if step is None:
            step = self.step_count
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step=None):
        """记录直方图数据"""
        if step is None:
            step = self.step_count
        self.writer.add_histogram(tag, values, step)

    def log_network(self, model, input_to_model=None, step=None):
        """记录网络结构和参数直方图"""
        if step is None:
            step = self.step_count

        # 记录模型参数的直方图
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"parameters/{name}", param.data, step)
            if param.grad is not None:
                self.writer.add_histogram(f"gradients/{name}", param.grad.data, step)

        # 如果提供了输入，也可以添加计算图
        if input_to_model is not None:
            self.writer.add_graph(model, input_to_model)

    def log_episode(self, rewards, loss, entropy=None, value_loss=None, policy_loss=None, step=None):
        """记录一个回合的数据"""
        if step is None:
            step = self.episode_count

        # 只有在达到记录频率时才记录
        if step % self.log_freq == 0:
            self.log_scalar("performance/episode_reward", sum(rewards), step)
            self.log_scalar("performance/episode_length", len(rewards), step)
            self.log_scalar("performance/mean_reward", sum(rewards) / len(rewards), step)
            self.log_scalar("losses/total_loss", loss, step)

            if entropy is not None:
                self.log_scalar("losses/entropy", entropy, step)
            if value_loss is not None:
                self.log_scalar("losses/value_loss", value_loss, step)
            if policy_loss is not None:
                self.log_scalar("losses/policy_loss", policy_loss, step)

            # 记录奖励直方图
            if step % self.histogram_freq == 0:
                self.log_histogram("rewards/distribution", torch.tensor(rewards), step)

        self.episode_count += 1

    def log_metrics(self, metrics_dict, step=None):
        """记录一组指标"""
        if step is None:
            step = self.step_count

        for key, value in metrics_dict.items():
            self.log_scalar(key, value, step)

        self.step_count += 1

    def close(self):
        """关闭日志记录器"""
        self.writer.close()