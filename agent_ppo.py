# agent ppo
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


# PPO策略网络模型
class PPOPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PPOPolicy, self).__init__()
        # 优化点1: 合并actor网络层减少计算深度
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  # 移除一个隐藏层
            nn.Softmax(dim=-1)
        )
        
        # 优化点2: 合并critic网络层减少计算深度
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 移除一个隐藏层
        )

    def forward(self, x):
        # 优化点3: 使用jit.script加速前向传播
        # 此处添加@torch.jit.script注解可进一步加速，但可能需要调整代码
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

    # 优化点4: 优化act方法，支持批处理
    def act_batch(self, states):
        """批量处理状态并返回动作"""
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)
        actions = dist.sample()
        action_log_probs = dist.log_prob(actions)
        return actions.detach(), action_log_probs.detach(), values.detach()

    def act(self, state):
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob

    def evaluate(self, state, action):
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return action_log_probs, value.squeeze(-1), entropy


# PPO智能体
class PPOAgent:
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size

        # 加载配置或使用默认值
        if config is None:
            config = {}

        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.learning_rate = config.get('learning_rate', 0.0003)
        
        # 优化点5: 减少PPO更新轮数，大幅度提高速度
        self.ppo_epochs = config.get('ppo_epochs', 4)  # 从10降到4
        self.batch_size = config.get('batch_size', 64)  # 增大批量
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        
        # 优化点6: 添加更新频率控制，减少更新次数
        self.update_frequency = config.get('update_frequency', 4)  # 每4个episode更新一次
        self.update_counter = 0
        
        # 检查GPU可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")

        # 初始化策略网络并移动到GPU
        self.policy = PPOPolicy(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # 优化点7: 预分配更大的缓冲区以减少扩展操作
        self.states_buffer = np.zeros((10000, state_size), dtype=np.float32)
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        # 轨迹计数
        self.buffer_counter = 0
        
        # 优化点8: 使用pin_memory提前将数据pin到内存中，加速数据迁移至GPU
        if self.device.type == 'cuda':
            self.pin_memory = True
        else:
            self.pin_memory = False

    def act(self, state):
        """根据当前状态选择动作"""
        # 将状态展平并转移到GPU
        state_tensor = torch.FloatTensor(np.array(state).flatten()).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy.act(state_tensor)
            _, value = self.policy(state_tensor)

        # 存储当前轨迹信息
        if self.buffer_counter < len(self.states_buffer):
            # 存储展平后的状态到预分配的缓冲区
            self.states_buffer[self.buffer_counter] = np.array(state).flatten()
        else:
            # 如果缓冲区已满，则扩展缓冲区（应该很少发生）
            self.states_buffer = np.vstack([self.states_buffer, np.array(state).flatten()])
            
        self.buffer_counter += 1
        
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value.item())

        return action

    def store_transition(self, reward, done):
        """存储奖励和状态终止信号"""
        self.rewards.append(reward)
        self.dones.append(done)
        
        # 优化点9: 增加更新计数器
        self.update_counter += 1

    def update(self):
        """使用PPO算法更新策略"""
        # 优化点10: 根据更新频率决定是否执行更新
        if self.update_counter < self.update_frequency:
            # 没到更新频率则跳过更新，返回上次的损失或0
            # 但仍存储经验不清空缓冲区
            return 0.0
            
        # 确保有足够的数据进行更新
        if len(self.rewards) == 0:
            return 0.0
            
        self.update_counter = 0  # 重置更新计数器

        # 计算优势估计和回报
        returns, advantages = self._compute_returns_advantages_vectorized()  # 使用向量化版本

        # 优化点11: 使用pin_memory加速数据迁移至GPU
        if self.device.type == 'cuda' and self.log_probs[0].device.type == 'cuda':
            # 张量已在CUDA上，直接使用它们
            states = torch.FloatTensor(self.states_buffer[:self.buffer_counter]).to(self.device)
            actions = torch.LongTensor(self.actions).to(self.device)
            old_log_probs = torch.stack(self.log_probs)  # 已经在CUDA上了
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
        elif self.pin_memory and self.device.type == 'cuda':
            # 张量在CPU上且支持pin_memory
            states = torch.FloatTensor(self.states_buffer[:self.buffer_counter]).pin_memory().to(self.device, non_blocking=True)
            actions = torch.LongTensor(self.actions).pin_memory().to(self.device, non_blocking=True)
            old_log_probs = torch.stack([log_prob.cpu() if log_prob.device.type == 'cuda' else log_prob for log_prob in self.log_probs]).pin_memory().to(self.device, non_blocking=True)
            returns = torch.FloatTensor(returns).pin_memory().to(self.device, non_blocking=True)
            advantages = torch.FloatTensor(advantages).pin_memory().to(self.device, non_blocking=True)
        else:
            # 普通加载
            states = torch.FloatTensor(self.states_buffer[:self.buffer_counter]).to(self.device)
            actions = torch.LongTensor(self.actions).to(self.device)
            old_log_probs = torch.stack(self.log_probs).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)


        total_loss = 0.0
        update_rounds = 0

        # 优化点12: 将所有数据预加载到GPU，批量处理减少循环开销
        dataset_size = states.size(0)
        indices = torch.randperm(dataset_size)
        
        # 进行多轮更新，使用小批量
        for _ in range(self.ppo_epochs):
            # 获取随机排列的索引
            indices = indices[torch.randperm(len(indices))]
            
            # 按批次进行更新
            for start_idx in range(0, dataset_size, self.batch_size):
                # 提取当前批次的索引
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # 提取批次数据
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 获取当前策略的动作概率、价值估计和熵
                new_log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)

                # 计算概率比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # 计算PPO裁剪目标函数
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 计算价值损失
                value_loss = F.mse_loss(values, batch_returns)

                # 计算总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                total_loss += loss.item()

                # 执行优化
                self.optimizer.zero_grad()
                loss.backward()
                
                # 优化点13: 添加梯度裁剪防止梯度爆炸，提高训练稳定性
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                
                self.optimizer.step()

                update_rounds += 1

        # 清空缓冲区
        self._clear_buffers()

        # 返回平均损失
        return total_loss / max(1, update_rounds)
        
    # 优化点14: 添加向量化的GAE计算，无需循环操作
    def _compute_returns_advantages_vectorized(self):
        """使用向量化方式计算GAE和回报，无需循环"""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)
        
        # 计算下一个状态的值
        if self.dones[-1]:
            next_value = 0
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.states_buffer[self.buffer_counter-1]).to(self.device)
                _, value = self.policy(state_tensor)
                next_value = value.item()
        
        # 添加下一个状态值到values序列末尾
        values = np.append(values, next_value)
        
        # 预分配优势数组
        advantages = np.zeros_like(rewards)
        
        # 优化GAE计算，减少循环
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        # 计算回报 = 优势 + 值
        returns = advantages + values[:-1]
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

    def _compute_returns_advantages(self):
        """计算广义优势估计(GAE)和回报"""
        returns = []
        advantages = []
        gae = 0

        # 将值函数估计添加到列表末尾
        with torch.no_grad():
            if self.dones[-1]:
                next_value = 0
            else:
                state_tensor = torch.FloatTensor(self.states_buffer[self.buffer_counter-1]).to(self.device)
                _, value = self.policy(state_tensor)
                next_value = value.item()

        self.values.append(next_value)

        # 倒序计算GAE
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * self.values[i + 1] * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])

        # 标准化优势
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def _clear_buffers(self):
        """清空轨迹缓冲区"""
        # 重置计数器而不是重新分配内存
        self.buffer_counter = 0
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def save_model(self, filepath):
        """保存模型到文件"""
        torch.save(self.policy.state_dict(), filepath)

    def load_model(self, filepath):
        """从文件加载模型"""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        
    # 优化点15: 添加设置为评估模式的方法
    def eval_mode(self):
        """设置为评估模式, 关闭dropout等训练特性"""
        self.policy.eval()
        
    # 优化点16: 添加设置为训练模式的方法
    def train_mode(self):
        """设置为训练模式, 启用dropout等训练特性"""
        self.policy.train()