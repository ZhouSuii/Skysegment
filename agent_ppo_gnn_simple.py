# 简化版 GNN-PPO：移除复杂设计，专注核心功能
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

# === 回滚到简化的GCN实现 ===
class SimpleGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleGCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        
        # 构建邻接矩阵（原始简洁实现）
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        if edge_index.size(1) > 0:  # 检查是否有边
            adj[edge_index[0], edge_index[1]] = 1.0
        
        # 添加自连接
        adj += torch.eye(num_nodes, device=x.device)
        
        # 度归一化
        degree = adj.sum(dim=1, keepdim=True)
        degree = torch.where(degree > 0, degree, torch.ones_like(degree))
        adj = adj / degree
        
        # 图卷积：A * X * W
        out = torch.mm(adj, x)
        out = self.linear(out)
        return out


# === 优化后的GNN策略网络：支持真正的批量处理 ===
class SimplePPOPolicyGNN(nn.Module):
    def __init__(self, node_feature_dim, action_size, hidden_dim=64):
        super(SimplePPOPolicyGNN, self).__init__()
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        
        # === 超简化：只用1层GNN ===
        self.gnn = SimpleGCNConv(node_feature_dim, hidden_dim)
        
        # === 直接动作预测（移除双头设计）===
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size),
            nn.Softmax(dim=-1)
        )
        
        # === 简化的Critic ===
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # === 优化：改进权重初始化 ===
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化，提升训练稳定性"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He初始化适用于ReLU激活
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, graph_data):
        x = graph_data['node_features']
        edge_index = graph_data['edge_index']
        
        # 单层GNN特征提取
        x = F.relu(self.gnn(x, edge_index))
        
        # 全局图表示：简单平均池化
        graph_repr = torch.mean(x, dim=0, keepdim=True)
        
        # 动作概率和价值
        action_probs = self.actor(graph_repr).squeeze(0)  # [action_size]
        value = self.critic(graph_repr).squeeze()  # scalar
        
        return action_probs, value

    def forward_batch(self, batch_graph_data_list):
        """=== 新增：真正的批量前向传播 ==="""
        batch_size = len(batch_graph_data_list)
        device = next(self.parameters()).device
        
        # 批量处理每个图
        batch_action_probs = []
        batch_values = []
        
        for graph_data in batch_graph_data_list:
            # 确保数据在正确设备上
            if isinstance(graph_data['node_features'], np.ndarray):
                x = torch.tensor(graph_data['node_features'], dtype=torch.float32, device=device)
            else:
                x = graph_data['node_features'].to(device)
                
            if isinstance(graph_data['edge_index'], np.ndarray):
                edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long, device=device)
            else:
                edge_index = graph_data['edge_index'].to(device)
            
            # GNN特征提取
            x = F.relu(self.gnn(x, edge_index))
            graph_repr = torch.mean(x, dim=0, keepdim=True)
            
            # 动作概率和价值
            action_probs = self.actor(graph_repr).squeeze(0)
            value = self.critic(graph_repr).squeeze()
            
            batch_action_probs.append(action_probs)
            batch_values.append(value)
        
        # 堆叠结果
        batch_action_probs = torch.stack(batch_action_probs)  # [batch_size, action_size]
        batch_values = torch.stack(batch_values)  # [batch_size]
        
        return batch_action_probs, batch_values

    def evaluate_batch(self, batch_graph_data_list, batch_actions):
        """=== 新增：批量评估 ==="""
        batch_action_probs, batch_values = self.forward_batch(batch_graph_data_list)
        
        # 计算动作对数概率
        action_dists = Categorical(batch_action_probs)
        action_log_probs = action_dists.log_prob(batch_actions)
        
        # 计算熵
        entropies = action_dists.entropy()
        
        return action_log_probs, batch_values, entropies

    def act(self, graph_data):
        action_probs, _ = self.forward(graph_data)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def evaluate(self, graph_data, action):
        action_probs, value = self.forward(graph_data)
        dist = Categorical(action_probs)
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action_log_prob, value, entropy


# === 大幅优化的PPO智能体：真正的批量处理 ===
class SimplePPOAgentGNN:
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size
        
        if config is None:
            config = {}

        # === 更保守的超参数 ===
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.learning_rate = config.get('learning_rate', 0.0003)  # 适中的学习率
        self.ppo_epochs = config.get('ppo_epochs', 4)  # 标准PPO轮数
        self.batch_size = config.get('batch_size', 128)  # 增大批量
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.update_frequency = config.get('update_frequency', 64)  # 更大的更新频率
        
        # 缓冲区
        self.memory_capacity = config.get('memory_capacity', 20000)
        self.graph_data_buffer = []
        self.actions = np.zeros(self.memory_capacity, dtype=np.int64)
        self.log_probs = np.zeros(self.memory_capacity, dtype=np.float32)
        self.rewards = np.zeros(self.memory_capacity, dtype=np.float32)
        self.dones = np.zeros(self.memory_capacity, dtype=np.float32)
        self.values = np.zeros(self.memory_capacity, dtype=np.float32)
        
        self.buffer_ptr = 0
        self.traj_start_ptr = 0
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 优化简化PPO-GNN使用设备: {self.device}")

        # === 简化的网络 ===
        self.policy = SimplePPOPolicyGNN(
            node_feature_dim=state_size,
            action_size=action_size,
            hidden_dim=config.get('hidden_dim', 128)  # 增加隐藏层大小
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # tensorboard
        from tensorboard_logger import TensorboardLogger
        tensorboard_config = config.get('tensorboard_config', {})
        self.use_tensorboard = config.get('use_tensorboard', True)
        if self.use_tensorboard:
            self.logger = TensorboardLogger(tensorboard_config)
        else:
            self.logger = None

        # === 新增：批量处理优化 ===
        self.enable_batch_processing = config.get('enable_batch_processing', True)
        print(f"🔧 批量处理: {'启用' if self.enable_batch_processing else '禁用'}")

    def _move_graph_to_device(self, graph_data, target_device=None):
        if target_device is None:
            target_device = self.device
        
        moved_data = {}
        for key, value in graph_data.items():
            if isinstance(value, torch.Tensor):
                moved_data[key] = value.to(target_device)
            elif isinstance(value, np.ndarray):
                moved_data[key] = torch.tensor(value).to(target_device)
            else:
                moved_data[key] = value
        return moved_data

    def act(self, graph_data):
        graph_data = self._move_graph_to_device(graph_data)
        
        self.policy.eval()
        with torch.no_grad():
            action, log_prob = self.policy.act(graph_data)
            _, value = self.policy(graph_data)
        self.policy.train()

        # 存储到缓冲区
        if self.buffer_ptr < self.memory_capacity:
            cpu_graph_data = self._move_graph_to_device(graph_data, target_device='cpu')
            self.graph_data_buffer.append(cpu_graph_data)
            
            self.actions[self.buffer_ptr] = action
            self.log_probs[self.buffer_ptr] = log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob
            self.values[self.buffer_ptr] = value.item()
        else:
            print("Warning: 简化PPO-GNN buffer overflow!")
            self.buffer_ptr = 0
            self.graph_data_buffer = []
            cpu_graph_data = self._move_graph_to_device(graph_data, target_device='cpu')
            self.graph_data_buffer.append(cpu_graph_data)
            
            self.actions[self.buffer_ptr] = action
            self.log_probs[self.buffer_ptr] = log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob
            self.values[self.buffer_ptr] = value.item()

        return action

    def store_transition(self, reward, done):
        if self.buffer_ptr < self.memory_capacity:
            self.rewards[self.buffer_ptr] = reward
            self.dones[self.buffer_ptr] = float(done)
            self.buffer_ptr += 1

    def update(self):
        steps_collected = self.buffer_ptr - self.traj_start_ptr
        
        if steps_collected < self.update_frequency and self.buffer_ptr < self.memory_capacity:
            return 0.0
            
        if steps_collected <= 0:
            self.traj_start_ptr = self.buffer_ptr
            return 0.0

        # 数据准备
        indices = np.arange(self.traj_start_ptr, self.buffer_ptr)
        graph_data_list = [self.graph_data_buffer[i] for i in indices]
        actions_np = self.actions[indices]
        old_log_probs_np = self.log_probs[indices]
        rewards_np = self.rewards[indices]
        dones_np = self.dones[indices]
        values_np = self.values[indices]

        # 计算优势和回报
        returns, advantages = self._compute_returns_advantages_vectorized(rewards_np, dones_np, values_np)

        # 转移到GPU
        actions = torch.tensor(actions_np, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(old_log_probs_np, dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        # === 核心优化：真正的批量PPO更新 ===
        total_loss = 0.0
        update_rounds = 0
        dataset_size = len(indices)
        current_batch_size = min(self.batch_size, dataset_size)

        for epoch in range(self.ppo_epochs):
            perm_indices = np.random.permutation(dataset_size)
            
            for start_idx in range(0, dataset_size, current_batch_size):
                end_idx = min(start_idx + current_batch_size, dataset_size)
                batch_indices = perm_indices[start_idx:end_idx]
                
                # === 关键优化：批量处理而不是逐个处理 ===
                if self.enable_batch_processing and len(batch_indices) > 1:
                    # 使用真正的批量处理
                    batch_graph_data_list = [graph_data_list[i] for i in batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns_tensor[batch_indices]
                    batch_advantages = advantages_tensor[batch_indices]
                    
                    # 批量评估
                    new_log_probs, values, entropies = self.policy.evaluate_batch(
                        batch_graph_data_list, batch_actions
                    )
                    
                    # PPO损失计算（向量化）
                    ratios = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_loss = F.mse_loss(values, batch_returns)
                    entropy_loss = -entropies.mean()
                    
                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                    
                    # 单次反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    update_rounds += 1
                else:
                    # 回退到逐个处理（小批量时）
                    batch_loss = 0.0
                    for i in batch_indices:
                        graph_data = self._move_graph_to_device(graph_data_list[i])
                        
                        new_log_prob, value, entropy = self.policy.evaluate(graph_data, actions[i])

                        # PPO损失计算
                        ratio = torch.exp(new_log_prob - old_log_probs[i])
                        surr1 = ratio * advantages_tensor[i]
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_tensor[i]
                        policy_loss = -torch.min(surr1, surr2)

                        value_loss = F.mse_loss(value, returns_tensor[i])
                        entropy_loss = -entropy

                        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                        batch_loss += loss

                    # 平均批次损失并反向传播
                    avg_batch_loss = batch_loss / len(batch_indices)
                    total_loss += avg_batch_loss.item()

                    self.optimizer.zero_grad()
                    avg_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    self.optimizer.step()

                    update_rounds += 1

        # 清理
        self.traj_start_ptr = self.buffer_ptr
        if self.buffer_ptr == self.memory_capacity:
            self.traj_start_ptr = 0
            self.buffer_ptr = 0
            self.graph_data_buffer = []

        return total_loss / max(1, update_rounds)

    def _compute_returns_advantages_vectorized(self, rewards, dones, values):
        """计算回报和优势"""
        if dones[-1]:
            next_value = 0.0
        else:
            with torch.no_grad():
                last_state_idx = self.buffer_ptr - 1
                if last_state_idx >= 0 and last_state_idx < len(self.graph_data_buffer):
                    last_graph_data = self._move_graph_to_device(self.graph_data_buffer[last_state_idx])
                    _, value = self.policy(last_graph_data)
                    next_value = value.item()
                else:
                    next_value = 0.0
        
        values_with_next = np.append(values, next_value)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_with_next[t + 1] * (1 - dones[t]) - values_with_next[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + values
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

    def save_model(self, filepath):
        torch.save(self.policy.state_dict(), filepath)
        if self.logger is not None:
            self.logger.close()

    def load_model(self, filepath):
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        
    def eval_mode(self):
        self.policy.eval()
        
    def train_mode(self):
        self.policy.train() 