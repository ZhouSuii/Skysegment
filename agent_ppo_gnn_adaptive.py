# 自适应GNN-PPO：根据图规模智能调整架构和训练策略
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import math

class AdaptiveSimpleGCNConv(nn.Module):
    """自适应图卷积层"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(AdaptiveSimpleGCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, edge_index, apply_dropout=True):
        num_nodes = x.size(0)
        
        # 构建邻接矩阵
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        if edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
        
        # 添加自连接
        adj += torch.eye(num_nodes, device=x.device)
        
        # 改进的度归一化：对称归一化
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.where(degree > 0, 1.0 / torch.sqrt(degree), torch.zeros_like(degree))
        degree_matrix = torch.diag(degree_inv_sqrt)
        adj_normalized = torch.mm(torch.mm(degree_matrix, adj), degree_matrix)
        
        # 图卷积：D^(-1/2) * A * D^(-1/2) * X * W
        out = torch.mm(adj_normalized, x)
        out = self.linear(out)
        
        if apply_dropout and self.training:
            out = self.dropout(out)
            
        return out


class AdaptivePPOPolicyGNN(nn.Module):
    """自适应GNN策略网络"""
    def __init__(self, node_feature_dim, action_size, num_nodes, config=None):
        super(AdaptivePPOPolicyGNN, self).__init__()
        self.action_size = action_size
        self.num_nodes = num_nodes
        
        # === 根据图规模自适应架构 ===
        if num_nodes <= 15:
            # 小图：轻量化架构
            self.hidden_dim = 32
            self.num_gnn_layers = 1
            self.dropout_rate = 0.0
            self.use_residual = False
        elif num_nodes <= 30:
            # 中图：平衡架构
            self.hidden_dim = 64
            self.num_gnn_layers = 2
            self.dropout_rate = 0.1
            self.use_residual = True
        else:
            # 大图：复杂架构
            self.hidden_dim = min(128, num_nodes * 2)  # 避免过度参数化
            self.num_gnn_layers = 2
            self.dropout_rate = 0.2
            self.use_residual = True
        
        print(f"📐 自适应架构：{num_nodes}节点 -> hidden_dim={self.hidden_dim}, layers={self.num_gnn_layers}")
        
        # GNN层
        self.gnn_layers = nn.ModuleList()
        for i in range(self.num_gnn_layers):
            in_dim = node_feature_dim if i == 0 else self.hidden_dim
            self.gnn_layers.append(
                AdaptiveSimpleGCNConv(in_dim, self.hidden_dim, self.dropout_rate)
            )
        
        # 残差连接的投影层
        if self.use_residual and node_feature_dim != self.hidden_dim:
            self.residual_proj = nn.Linear(node_feature_dim, self.hidden_dim)
        
        # === 改进的Actor ===
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, action_size),
            nn.Softmax(dim=-1)
        )
        
        # === 改进的Critic：层归一化 ===
        self.critic = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He初始化用于ReLU激活
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, graph_data):
        x = graph_data['node_features']
        edge_index = graph_data['edge_index']
        
        # 保存原始特征用于残差连接
        if self.use_residual:
            if hasattr(self, 'residual_proj'):
                residual = self.residual_proj(x)
            else:
                residual = x
        
        # GNN特征提取
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index, apply_dropout=(i < len(self.gnn_layers) - 1))
            x = F.relu(x)
            
            # 添加残差连接（仅在第一层）
            if i == 0 and self.use_residual:
                x = x + residual
        
        # === 改进的图级别表示：加权平均池化 ===
        # 计算节点重要性权重
        node_importance = torch.softmax(torch.sum(x, dim=1), dim=0)  # [num_nodes]
        graph_repr = torch.sum(x * node_importance.unsqueeze(1), dim=0, keepdim=True)  # [1, hidden_dim]
        
        # 动作概率和价值
        action_probs = self.actor(graph_repr).squeeze(0)  # [action_size]
        value = self.critic(graph_repr).squeeze()  # scalar
        
        return action_probs, value
    
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


class AdaptivePPOAgentGNN:
    """自适应PPO智能体"""
    def __init__(self, state_size, action_size, num_nodes, config=None):
        self.state_size = state_size
        self.action_size = action_size
        self.num_nodes = num_nodes
        
        if config is None:
            config = {}
        
        # === 根据图规模自适应超参数 ===
        if num_nodes <= 15:
            # 小图：更快学习，更少探索
            self.learning_rate = config.get('learning_rate', 0.001)
            self.ppo_epochs = config.get('ppo_epochs', 4)
            self.batch_size = config.get('batch_size', min(16, max(8, num_nodes)))
            self.entropy_coef = config.get('entropy_coef', 0.05)
            self.update_frequency = config.get('update_frequency', 4)
        elif num_nodes <= 30:
            # 中图：平衡设置
            self.learning_rate = config.get('learning_rate', 0.0003)
            self.ppo_epochs = config.get('ppo_epochs', 3)
            self.batch_size = config.get('batch_size', min(32, max(16, num_nodes)))
            self.entropy_coef = config.get('entropy_coef', 0.02)
            self.update_frequency = config.get('update_frequency', 6)
        else:
            # 大图：更保守学习，更多探索
            self.learning_rate = config.get('learning_rate', 0.0001)
            self.ppo_epochs = config.get('ppo_epochs', 2)
            self.batch_size = config.get('batch_size', min(64, max(32, num_nodes)))
            self.entropy_coef = config.get('entropy_coef', 0.01)
            self.update_frequency = config.get('update_frequency', 8)
        
        print(f"🎛️ 自适应超参数：lr={self.learning_rate}, batch={self.batch_size}, entropy={self.entropy_coef}")
        
        # 固定超参数
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        
        # 缓冲区
        self.memory_capacity = config.get('memory_capacity', min(20000, num_nodes * 500))
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
        print(f"🔧 自适应PPO-GNN使用设备: {self.device}")
        
        # 网络初始化
        self.policy = AdaptivePPOPolicyGNN(
            node_feature_dim=state_size,
            action_size=action_size,
            num_nodes=num_nodes,
            config=config
        ).to(self.device)
        
        # === 学习率调度器 ===
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=50, verbose=True
        )
        
        # tensorboard (简化版本不使用)
        self.logger = None
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
    
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
            # Buffer reset
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
        
        # === 改进的PPO更新：早停机制 ===
        total_loss = 0.0
        update_rounds = 0
        dataset_size = len(indices)
        current_batch_size = min(self.batch_size, dataset_size)
        
        best_loss = float('inf')
        patience_counter = 0
        patience_limit = max(1, self.ppo_epochs // 2)
        
        for epoch in range(self.ppo_epochs):
            perm_indices = np.random.permutation(dataset_size)
            epoch_loss = 0.0
            epoch_rounds = 0
            
            for start_idx in range(0, dataset_size, current_batch_size):
                end_idx = min(start_idx + current_batch_size, dataset_size)
                batch_indices = perm_indices[start_idx:end_idx]
                
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
                
                # 反向传播
                avg_batch_loss = batch_loss / len(batch_indices)
                
                self.optimizer.zero_grad()
                avg_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                epoch_loss += avg_batch_loss.item()
                epoch_rounds += 1
            
            # 早停检查
            avg_epoch_loss = epoch_loss / max(1, epoch_rounds)
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    break
            
            total_loss += epoch_loss
            update_rounds += epoch_rounds
        
        # 学习率调度
        avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0
        self.scheduler.step(avg_reward)
        
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