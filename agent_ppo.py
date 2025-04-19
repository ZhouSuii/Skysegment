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
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

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
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.batch_size = config.get('batch_size', 32)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)

        # 初始化策略网络
        self.policy = PPOPolicy(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # 用于存储轨迹
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def act(self, state):
        """根据当前状态选择动作"""
        state_tensor = torch.FloatTensor(state).flatten()
        with torch.no_grad():
            action, log_prob = self.policy.act(state_tensor)
            _, value = self.policy(state_tensor)

        # 存储当前轨迹信息
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value.item())

        return action

    def store_transition(self, reward, done):
        """存储奖励和状态终止信号"""
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self):
        """使用PPO算法更新策略"""
        # 确保有足够的数据进行更新
        if len(self.rewards) == 0:
            return 0.0

        # 计算优势估计和回报
        returns, advantages = self._compute_returns_advantages()

        # 转换为张量
        states = torch.FloatTensor([s.flatten() for s in self.states])
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        total_loss = 0.0
        update_rounds = 0

        # 进行多轮更新
        for _ in range(self.ppo_epochs):
            # 按批次进行更新
            for idx in range(0, len(self.states), self.batch_size):
                batch_idx = slice(idx, min(idx + self.batch_size, len(self.states)))

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # 获取当前策略的动作概率、价值估计和熵
                new_log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)

                # 计算概率比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs.detach())

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
                self.optimizer.step()

                update_rounds += 1

        # 清空缓冲区
        self._clear_buffers()

        # 返回平均损失
        return total_loss / max(1, update_rounds)

    def _compute_returns_advantages(self):
        """计算广义优势估计(GAE)和回报"""
        returns = []
        advantages = []
        gae = 0

        # 将值函数估计添加到列表末尾
        with torch.no_grad():
            next_value = 0 if self.dones[-1] else self.policy(torch.FloatTensor(self.states[-1]).flatten())[1].item()

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
        self.states = []
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
        self.policy.load_state_dict(torch.load(filepath))