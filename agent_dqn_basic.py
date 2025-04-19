# dqn agent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# DQN网络模型
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        # 加载配置或使用默认值
        if config is None:
            config = {}
        self.gamma = config.get('gamma', 0.95)  # 折扣因子
        self.epsilon = config.get('epsilon', 1.0)  # 初始探索率
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.target_update_freq = config.get('target_update_freq', 10)

        # 初始化模型
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

        # 训练计数器
        self.train_count = 0

    def update_target_model(self):
        """更新目标网络的参数"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """根据当前状态选择动作"""
        # 探索：随机选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # 利用：使用模型预测最佳动作
        state_tensor = torch.FloatTensor(state).flatten()
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        """从经验回放中学习"""
        if len(self.memory) < batch_size:
            return 0.0  # 返回默认loss值

        total_loss = 0.0
        # 从记忆中随机采样
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).flatten()
            next_state_tensor = torch.FloatTensor(next_state).flatten()

            # 计算目标 Q 值
            if done:
                target = reward
            else:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()

            # 获取当前 Q 值预测
            current_q = self.model(state_tensor)
            target_f = current_q.clone()
            target_f[action] = target

            # 执行优化步骤
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(current_q, target_f.detach())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # 探索率衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 更新训练计数器并检查是否需要更新目标网络
        self.train_count += 1
        if self.train_count % self.target_update_freq == 0:
            self.update_target_model()

        return total_loss / batch_size  # 返回平均损失

    def save_model(self, filepath):
        """保存模型到文件"""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """从文件加载模型"""
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(torch.load(filepath))