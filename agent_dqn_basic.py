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
        
        # 加载配置或使用默认值
        if config is None:
            config = {}
        self.gamma = config.get('gamma', 0.95)  # 折扣因子
        self.epsilon = config.get('epsilon', 1.0)  # 初始探索率
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.target_update_freq = config.get('target_update_freq', 10)
        self.memory_capacity = config.get('memory_capacity', 2000)
        
        # 检查是否有可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            
        # 初始化经验回放缓冲区（使用更高效的数组存储）
        self.memory_counter = 0
        self.memory = {
            'state': np.zeros((self.memory_capacity, self.state_size), dtype=np.float32),
            'action': np.zeros(self.memory_capacity, dtype=np.int64),
            'reward': np.zeros(self.memory_capacity, dtype=np.float32),
            'next_state': np.zeros((self.memory_capacity, self.state_size), dtype=np.float32),
            'done': np.zeros(self.memory_capacity, dtype=np.float32)
        }

        # 初始化模型并移动到GPU
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

        # 训练计数器
        self.train_count = 0

    def update_target_model(self):
        """更新目标网络的参数"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        index = self.memory_counter % self.memory_capacity
        
        # 平坦化状态以匹配缓冲区形状
        flat_state = np.array(state).flatten()
        flat_next_state = np.array(next_state).flatten()
        
        # 存储到缓冲区
        self.memory['state'][index] = flat_state
        self.memory['action'][index] = action
        self.memory['reward'][index] = reward
        self.memory['next_state'][index] = flat_next_state
        self.memory['done'][index] = float(done)
        
        self.memory_counter += 1

    def act(self, state):
        """根据当前状态选择动作"""
        # 探索：随机选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # 利用：使用模型预测最佳动作
        state_tensor = torch.FloatTensor(np.array(state).flatten()).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        """从经验回放中学习，使用真正的批处理"""
        # 确保有足够的样本
        memory_size = min(self.memory_counter, self.memory_capacity)
        if memory_size < batch_size:
            return 0.0

        # 随机采样批量索引
        indices = np.random.choice(memory_size, batch_size, replace=False)
        
        # 获取批量数据并直接移至GPU
        batch_states = torch.FloatTensor(self.memory['state'][indices]).to(self.device)
        batch_actions = torch.LongTensor(self.memory['action'][indices]).to(self.device)
        batch_rewards = torch.FloatTensor(self.memory['reward'][indices]).to(self.device)
        batch_next_states = torch.FloatTensor(self.memory['next_state'][indices]).to(self.device)
        batch_dones = torch.FloatTensor(self.memory['done'][indices]).to(self.device)

        # 批量计算当前Q值
        current_q_values = self.model(batch_states)
        current_q_values_selected = current_q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        
        # 批量计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_model(batch_next_states)
            max_next_q_values = next_q_values.max(1)[0]
            expected_q_values = batch_rewards + (1 - batch_dones) * self.gamma * max_next_q_values
        
        # 计算损失并优化
        loss = nn.MSELoss()(current_q_values_selected, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 探索率衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 更新训练计数器并检查是否需要更新目标网络
        self.train_count += 1
        if self.train_count % self.target_update_freq == 0:
            self.update_target_model()

        return loss.item()

    def save_model(self, filepath):
        """保存模型到文件"""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """从文件加载模型"""
        state_dict = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.target_model.load_state_dict(state_dict)