import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# DQN网络模型
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=[256, 256]):
        super(DQN, self).__init__()
        
        # 创建动态网络架构
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            # 添加BatchNorm来改善训练稳定性
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # 添加输出层
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


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
        self.hidden_sizes = config.get('hidden_sizes', [256, 256])
        self.batch_size = config.get('batch_size', 512)
        self.jit_compile = config.get('jit_compile', True)
        
        # 检查是否有可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            torch.backends.cudnn.benchmark = True
            
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
        self.model = DQN(state_size, action_size, self.hidden_sizes).to(self.device)
        self.target_model = DQN(state_size, action_size, self.hidden_sizes).to(self.device)
        
        # 使用JIT编译提高GPU性能
        if self.jit_compile and torch.__version__ >= '1.8.0':
            self.model = torch.jit.script(self.model)
            self.target_model = torch.jit.script(self.target_model)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

        # 训练计数器
        self.train_count = 0

        # 预先创建CUDA张量以减少内存分配
        self.cuda_batches = {
            'state': torch.zeros((self.batch_size, self.state_size), device=self.device, dtype=torch.float32),
            'action': torch.zeros((self.batch_size), device=self.device, dtype=torch.int64),
            'reward': torch.zeros((self.batch_size), device=self.device, dtype=torch.float32),
            'next_state': torch.zeros((self.batch_size, self.state_size), device=self.device, dtype=torch.float32),
            'done': torch.zeros((self.batch_size), device=self.device, dtype=torch.float32),
        }
        # 添加TensorBoard支持
        from tensorboard_logger import TensorboardLogger
        tensorboard_config = config.get('tensorboard_config', {})
        self.use_tensorboard = config.get('use_tensorboard', True)
        if self.use_tensorboard:
            self.logger = TensorboardLogger(tensorboard_config)
            self.logger.experiment_name = "dqn_agent"
        else:
            self.logger = None

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

    def act(self, state, action_mask=None):
        """根据当前状态选择动作，支持动作掩码"""
        # 探索：随机选择动作
        if np.random.rand() <= self.epsilon:
            if action_mask is not None:
                # 从有效动作中随机选择
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions)
            return random.randrange(self.action_size)

        # 利用：使用模型预测最佳动作
        state_array = np.array(state).flatten()
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)

        # 将模型设置为评估模式
        self.model.eval()

        with torch.no_grad():
            act_values = self.model(state_tensor)
            
            # 如果有动作掩码，将无效动作设为极小值
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
                if mask_tensor.shape != act_values.shape:
                     print(f"警告: act_values shape {act_values.shape} 与 mask_tensor shape {mask_tensor.shape} 不匹配！")
                     # 尝试调整掩码形状，但这可能指示更深层的问题
                     mask_tensor = mask_tensor.view(act_values.shape)
    
                invalid_mask = 1.0 - mask_tensor
                act_values = act_values - invalid_mask * 1e9
        
        self.model.train()

        return torch.argmax(act_values).item()

    def replay(self, batch_size=None):
        """从经验回放中学习，使用优化的批处理"""
        if batch_size is None:
            batch_size = self.batch_size
            
        # 确保有足够的样本
        memory_size = min(self.memory_counter, self.memory_capacity)
        if memory_size < batch_size:
            return 0.0

        # 随机采样批量索引
        indices = np.random.choice(memory_size, batch_size, replace=False)
        
        # 使用预先分配的CUDA张量
        self.cuda_batches['state'].copy_(torch.FloatTensor(self.memory['state'][indices]))
        self.cuda_batches['action'].copy_(torch.LongTensor(self.memory['action'][indices]))
        self.cuda_batches['reward'].copy_(torch.FloatTensor(self.memory['reward'][indices]))
        self.cuda_batches['next_state'].copy_(torch.FloatTensor(self.memory['next_state'][indices]))
        self.cuda_batches['done'].copy_(torch.FloatTensor(self.memory['done'][indices]))
        
        # 批量计算当前Q值
        current_q_values = self.model(self.cuda_batches['state'])
        current_q_values_selected = current_q_values.gather(1, self.cuda_batches['action'].unsqueeze(1)).squeeze(1)
        
        # 批量计算目标Q值 (实现Double DQN)
        with torch.no_grad():
            # 1. 使用在线网络(self.model)为下一状态选择最佳动作
            next_q_values_online = self.model(self.cuda_batches['next_state'])
            best_next_actions = next_q_values_online.argmax(1)

            # 2. 使用目标网络(self.target_model)评估被选中动作的价值
            next_q_values_target = self.target_model(self.cuda_batches['next_state'])
            # 使用 .gather() 从目标网络中提取Q值
            q_value_of_best_action = next_q_values_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            # 3. 计算最终的目标Q值
            expected_q_values = self.cuda_batches['reward'] + (1 - self.cuda_batches['done']) * self.gamma * q_value_of_best_action
        
        # 计算损失并优化
        loss = nn.MSELoss()(current_q_values_selected, expected_q_values)

        # 在损失计算后记录
        if self.logger is not None and self.train_count % self.logger.log_freq == 0:
            self.logger.log_scalar("losses/q_loss", loss.item(), self.train_count)
            self.logger.log_scalar("dqn/epsilon", self.epsilon, self.train_count)

            # 记录Q值分布
            if self.train_count % self.logger.histogram_freq == 0:
                self.logger.log_histogram("q_values/distribution", current_q_values.detach(), self.train_count)
                self.logger.log_network(self.model)

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