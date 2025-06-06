# agent ppo with GNN improvements
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

# === GNN模块新增导入 ===
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    print("警告: torch_geometric未安装，将使用简化的图神经网络实现")
    HAS_TORCH_GEOMETRIC = False


# === 新增：简化的GCN实现（如果torch_geometric不可用） ===
class SimpleGCNConv(nn.Module):
    """简化版本的图卷积层，不依赖torch_geometric"""
    def __init__(self, in_channels, out_channels):
        super(SimpleGCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        num_nodes = x.size(0)
        
        # 构建邻接矩阵
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
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


# === 修改：PPO策略网络模型，添加GNN模块 ===
class PPOPolicyGNN(nn.Module):
    def __init__(self, node_feature_dim, num_partitions, hidden_dim=128, gnn_layers=2):
        super(PPOPolicyGNN, self).__init__()
        self.num_partitions = num_partitions
        self.hidden_dim = hidden_dim
        
        # === 新增：GNN特征提取模块 ===
        if HAS_TORCH_GEOMETRIC:
            self.gnn_layers = nn.ModuleList([
                GCNConv(node_feature_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(gnn_layers)
            ])
        else:
            self.gnn_layers = nn.ModuleList([
                SimpleGCNConv(node_feature_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(gnn_layers)
            ])
        
        self.gnn_dropout = nn.Dropout(0.1)
        
        # === 修改：Actor改为双头设计 ===
        # 头1：节点选择 (选择哪个节点进行重分配)
        self.node_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 每个节点的选择概率
        )
        
        # 头2：分区分配 (将选中节点分配到哪个分区)
        self.partition_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_partitions)
        )
        
        # === 修改：Critic网络使用全局图表示 ===
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, graph_data):
        """
        === 修改：输入从矩阵改为图数据 ===
        graph_data: dict包含
            - node_features: [num_nodes, feature_dim]
            - edge_index: [2, num_edges]
            - current_partition: [num_nodes] 当前分区分配
        """
        x = graph_data['node_features']  # [num_nodes, feature_dim]
        edge_index = graph_data['edge_index']  # [2, num_edges]
        current_partition = graph_data['current_partition']  # [num_nodes]
        
        # === 新增：GNN特征提取 ===
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)
            if i < len(self.gnn_layers) - 1:  # 最后一层不加激活
                x = F.relu(x)
                x = self.gnn_dropout(x)
        
        # x现在是[num_nodes, hidden_dim]的节点嵌入
        
        # === 新增：双头Actor设计 ===
        # 节点选择概率
        node_logits = self.node_selector(x).squeeze(-1)  # [num_nodes]
        
        # 分区选择概率 (对每个节点)
        partition_logits = self.partition_selector(x)  # [num_nodes, num_partitions]
        
        # === 新增：无效动作屏蔽 ===
        # 屏蔽会导致分区为空的动作
        masked_partition_logits = self._apply_action_mask(
            partition_logits, current_partition
        )
        
        # === 修改：Critic使用图级别表示 ===
        # 简单的图级别聚合：平均池化
        graph_representation = torch.mean(x, dim=0, keepdim=True)  # [1, hidden_dim]
        value = self.critic(graph_representation)  # [1, 1]
        
        return {
            'node_logits': node_logits,
            'partition_logits': masked_partition_logits,
            'value': value.squeeze()
        }

    def _apply_action_mask(self, partition_logits, current_partition):
        """
        === 新增：无效动作屏蔽 ===
        屏蔽会导致某个分区变为空的动作
        """
        num_nodes, num_partitions = partition_logits.shape
        masked_logits = partition_logits.clone()
        
        # 计算每个分区的节点数
        partition_counts = torch.bincount(current_partition, minlength=num_partitions)
        
        for node_idx in range(num_nodes):
            current_part = current_partition[node_idx]
            # 如果当前分区只有这一个节点，不能移动到其他分区
            if partition_counts[current_part] <= 1:
                # 屏蔽所有其他分区，只能保持当前分区
                mask = torch.ones(num_partitions, dtype=torch.bool, device=partition_logits.device)
                mask[current_part] = False
                masked_logits[node_idx][mask] = float('-inf')
        
        return masked_logits

    def act_batch(self, graph_batch):
        """=== 修改：批量处理图数据 ==="""
        # 这里需要根据实际批处理需求实现
        # 目前先保持简单，单个图处理
        return self.act(graph_batch)

    def act(self, graph_data):
        """=== 修改：基于图数据的动作选择 ==="""
        outputs = self.forward(graph_data)
        
        # 节点选择
        node_dist = Categorical(logits=outputs['node_logits'])
        selected_node = node_dist.sample()
        node_log_prob = node_dist.log_prob(selected_node)
        
        # 分区选择 (针对选中的节点)
        partition_dist = Categorical(logits=outputs['partition_logits'][selected_node])
        selected_partition = partition_dist.sample()
        partition_log_prob = partition_dist.log_prob(selected_partition)
        
        # 组合动作和对数概率
        # action编码：node_idx * num_partitions + partition_idx (保持与原版兼容)
        action = selected_node * self.num_partitions + selected_partition
        total_log_prob = node_log_prob + partition_log_prob
        
        return action.item(), total_log_prob

    def evaluate(self, graph_data, action):
        """=== 修改：基于图数据的策略评估 ==="""
        outputs = self.forward(graph_data)
        
        # 解码动作
        node_idx = action // self.num_partitions
        partition_idx = action % self.num_partitions
        
        # 计算动作概率
        node_dist = Categorical(logits=outputs['node_logits'])
        node_log_prob = node_dist.log_prob(node_idx)
        
        partition_dist = Categorical(logits=outputs['partition_logits'][node_idx])
        partition_log_prob = partition_dist.log_prob(partition_idx)
        
        total_log_prob = node_log_prob + partition_log_prob
        
        # 计算熵
        node_entropy = node_dist.entropy()
        partition_entropy = partition_dist.entropy().mean()  # 所有节点的平均熵
        total_entropy = node_entropy + partition_entropy
        
        return total_log_prob, outputs['value'], total_entropy


# === 修改：PPO智能体类，适配GNN ===
class PPOAgentGNN:
    def __init__(self, state_size, action_size, config=None):
        # === 新增：图相关参数 ===
        self.state_size = state_size  # 现在表示节点特征维度
        self.action_size = action_size
        self.num_partitions = config.get('num_partitions', 2) if config else 2
        
        # 加载配置或使用默认值
        if config is None:
            config = {}

        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.learning_rate = config.get('learning_rate', 0.0003)
        
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.batch_size = config.get('batch_size', 64)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.update_frequency = config.get('update_frequency', 4)
        
        # === 修改：缓冲区存储图数据 ===
        self.memory_capacity = config.get('memory_capacity', 20000)
        # 存储图数据而不是扁平状态
        self.graph_data_buffer = []
        self.actions = np.zeros(self.memory_capacity, dtype=np.int64)
        self.log_probs = np.zeros(self.memory_capacity, dtype=np.float32)
        self.rewards = np.zeros(self.memory_capacity, dtype=np.float32)
        self.dones = np.zeros(self.memory_capacity, dtype=np.float32)
        self.values = np.zeros(self.memory_capacity, dtype=np.float32)
        
        self.buffer_ptr = 0
        self.traj_start_ptr = 0
        
        # 检查GPU可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO-GNN使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")

        # === 修改：初始化GNN策略网络 ===
        self.policy = PPOPolicyGNN(
            node_feature_dim=state_size,  # 节点特征维度
            num_partitions=self.num_partitions,
            hidden_dim=config.get('hidden_dim', 128),
            gnn_layers=config.get('gnn_layers', 2)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        if self.device.type == 'cuda':
            self.pin_memory = True
        else:
            self.pin_memory = False

        # tensorboard
        from tensorboard_logger import TensorboardLogger
        tensorboard_config = config.get('tensorboard_config', {})
        self.use_tensorboard = config.get('use_tensorboard', True)
        if self.use_tensorboard:
            self.logger = TensorboardLogger(tensorboard_config)
        else:
            self.logger = None

    def act(self, graph_data):
        """=== 修改：基于图数据的动作选择 ==="""
        # 确保图数据在正确的设备上
        graph_data = self._move_graph_to_device(graph_data)
        
        self.policy.eval()
        with torch.no_grad():
            action, log_prob = self.policy.act(graph_data)
            outputs = self.policy(graph_data)
            value = outputs['value']
        self.policy.train()

        # === 修改：存储图数据到缓冲区 ===
        if self.buffer_ptr < self.memory_capacity:
            # 存储图数据的CPU版本
            cpu_graph_data = self._move_graph_to_device(graph_data, target_device='cpu')
            self.graph_data_buffer.append(cpu_graph_data)
            
            self.actions[self.buffer_ptr] = action
            self.log_probs[self.buffer_ptr] = log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob
            self.values[self.buffer_ptr] = value.item()
        else:
            print("Warning: PPO-GNN buffer overflow!")
            self.buffer_ptr = 0
            # 重置图数据缓冲区
            self.graph_data_buffer = []
            cpu_graph_data = self._move_graph_to_device(graph_data, target_device='cpu')
            self.graph_data_buffer.append(cpu_graph_data)
            
            self.actions[self.buffer_ptr] = action
            self.log_probs[self.buffer_ptr] = log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob
            self.values[self.buffer_ptr] = value.item()

        return action

    def _move_graph_to_device(self, graph_data, target_device=None):
        """=== 新增：将图数据移动到指定设备 ==="""
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

    def store_transition(self, reward, done):
        """存储奖励和状态终止信号"""
        if self.buffer_ptr < self.memory_capacity:
            self.rewards[self.buffer_ptr] = reward
            self.dones[self.buffer_ptr] = float(done)
            self.buffer_ptr += 1

    def update(self):
        """=== 修改：使用图数据进行PPO更新 ==="""
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
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        # PPO更新循环
        total_loss = 0.0
        update_rounds = 0
        dataset_size = len(indices)
        current_batch_size = min(self.batch_size, dataset_size)

        for _ in range(self.ppo_epochs):
            perm_indices = np.random.permutation(dataset_size)
            
            for start_idx in range(0, dataset_size, current_batch_size):
                end_idx = min(start_idx + current_batch_size, dataset_size)
                batch_indices = perm_indices[start_idx:end_idx]
                
                # === 修改：处理图数据批次 ===
                batch_graph_data_list = [graph_data_list[i] for i in batch_indices]
                # 这里可以优化为真正的批处理，目前简化处理
                
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 计算批次损失
                batch_loss = 0.0
                for i, graph_data in enumerate(batch_graph_data_list):
                    graph_data = self._move_graph_to_device(graph_data)
                    
                    new_log_prob, value, entropy = self.policy.evaluate(
                        graph_data, batch_actions[i]
                    )

                    # PPO损失计算
                    ratio = torch.exp(new_log_prob - batch_old_log_probs[i])
                    surr1 = ratio * batch_advantages[i]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages[i]
                    policy_loss = -torch.min(surr1, surr2)

                    value_loss = F.mse_loss(value, batch_returns[i])
                    entropy_loss = -entropy

                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                    batch_loss += loss

                # 平均批次损失并反向传播
                avg_batch_loss = batch_loss / len(batch_graph_data_list)
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
            print("PPO-GNN buffer wrapped around.")

        return total_loss / max(1, update_rounds)
        
    def _compute_returns_advantages_vectorized(self, rewards, dones, values):
        """计算回报和优势 (保持原有逻辑)"""
        if dones[-1]:
            next_value = 0.0
        else:
            with torch.no_grad():
                last_state_idx = self.buffer_ptr - 1
                if last_state_idx < 0: 
                    last_state_idx = self.memory_capacity - 1
                
                # === 修改：使用图数据获取最后状态的值 ===
                if last_state_idx < len(self.graph_data_buffer):
                    last_graph_data = self._move_graph_to_device(self.graph_data_buffer[last_state_idx])
                    outputs = self.policy(last_graph_data)
                    next_value = outputs['value'].item()
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

    def _clear_buffers(self):
        """清空缓冲区"""
        self.buffer_ptr = 0
        self.traj_start_ptr = 0
        self.graph_data_buffer = []

    def save_model(self, filepath):
        """保存模型到文件"""
        torch.save(self.policy.state_dict(), filepath)
        if self.logger is not None:
            self.logger.close()

    def load_model(self, filepath):
        """从文件加载模型"""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        
    def eval_mode(self):
        """设置为评估模式"""
        self.policy.eval()
        
    def train_mode(self):
        """设置为训练模式"""
        self.policy.train()
