# gnn agent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import Data, Batch
import networkx as nx
from torch.distributions import Categorical

# 增强的GNN-DQN网络模型 - 添加了现代技术
class GNNDQN(nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim, num_layers=2, dropout_rate=0.1):
        super(GNNDQN, self).__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # 定义GNN层、BatchNorm和Dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 输入层
        self.convs.append(GCNConv(node_features, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Q值预测层
        self.q_prediction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index):
        # 保留原始输入用于可能的残差连接
        identity = x
        
        # 应用GNN层与BatchNorm、Dropout和残差连接
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # 记住当前状态用于残差连接
            x_res = x
            
            # GCN卷积
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # 添加残差连接(从第二层开始，且维度匹配时)
            if i > 0 and x_res.shape[-1] == x.shape[-1]:
                x = x + x_res
        
        # 节点级Q值预测
        q_values = self.q_prediction(x)
        
        return q_values


class GNNDQNAgent:
    def __init__(self, graph, num_partitions, config=None):
        self.graph = graph
        self.num_partitions = num_partitions
        self.num_nodes = len(graph.nodes())
        
        # 节点特征维度: 当前分区分配(one-hot) + 节点权重 + 节点度
        self.node_features = num_partitions + 2
        # 动作空间大小: 每个节点可以移动到任何一个分区
        self.action_size = self.num_nodes * num_partitions
        
        # 加载配置或使用默认值
        if config is None:
            config = {}
        self.gamma = config.get('gamma', 0.95)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.target_update_freq = config.get('target_update_freq', 10)
        self.memory_capacity = config.get('memory_size', 2000)
        self.batch_size = config.get('batch_size', 512)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.adam_beta1 = config.get('adam_beta1', 0.9)
        self.adam_beta2 = config.get('adam_beta2', 0.999)
        
        # 检查GPU可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GNN-DQN使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")
            # 优化CUDA性能
            torch.backends.cudnn.benchmark = True
        
        # 初始化经验回放缓冲区 (优化存储方式)
        self.memory_counter = 0
        self.current_memory_size = 0
        self.memory = {
            'partition_assignments': np.zeros((self.memory_capacity, self.num_nodes), dtype=np.int32),
            'actions': np.zeros(self.memory_capacity, dtype=np.int64),
            'rewards': np.zeros(self.memory_capacity, dtype=np.float32),
            'next_partition_assignments': np.zeros((self.memory_capacity, self.num_nodes), dtype=np.int32),
            'dones': np.zeros(self.memory_capacity, dtype=np.float32)
        }
        
        # 初始化GNN模型并移动到GPU
        self.model = GNNDQN(self.node_features, self.hidden_dim, num_partitions, 
                            self.num_layers, self.dropout_rate).to(self.device)
        self.target_model = GNNDQN(self.node_features, self.hidden_dim, num_partitions,
                                  self.num_layers, self.dropout_rate).to(self.device)
        
        # 预构建图结构
        self._build_graph_structure()
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2)
        )
        self.update_target_model()
        
        # 训练计数器
        self.train_count = 0
    
    def _build_graph_structure(self):
        """构建并预处理图结构"""
        # 创建边索引
        edge_list = list(self.graph.edges())
        # 转换为PyG格式的边索引
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # 添加反向边，确保是无向图
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        self.edge_index = edge_index.to(self.device)
        
        # 提取并归一化节点权重 (NumPy)
        node_weights_np = np.array([self.graph.nodes[i].get('weight', 1.0)
                                     for i in range(self.num_nodes)], dtype=np.float32)
        max_weight = np.max(node_weights_np) if len(node_weights_np) > 0 else 1.0
        node_weights_np = node_weights_np / (max_weight + 1e-8)  # 避免除零
        
        # 提取并归一化节点度 (NumPy)
        degrees_np = np.array([self.graph.degree[i]
                           for i in range(self.num_nodes)], dtype=np.float32)
        max_degree = np.max(degrees_np) if len(degrees_np) > 0 else 1.0
        degrees_np = degrees_np / (max_degree + 1e-8)  # 避免除零

        # Store fixed features on GPU for direct use in batch creation
        self.fixed_features_tensor = torch.tensor(
            np.column_stack((node_weights_np, degrees_np)), 
            dtype=torch.float32, device=self.device
        )
        # Keep NumPy version for single state conversion in act
        self.fixed_features_np = np.column_stack((node_weights_np, degrees_np))

    def _partition_to_features_batch(self, partition_assignments_np):
        """将一批分区分配 (NumPy) 转换为 GPU 上的节点特征张量"""
        batch_size = partition_assignments_np.shape[0]
        num_nodes = partition_assignments_np.shape[1]
        
        # 在 GPU 上预分配批处理特征张量
        # 形状: (batch_size * num_nodes, node_features)
        batch_features = torch.zeros((batch_size * num_nodes, self.node_features), 
                                   dtype=torch.float32, device=self.device)

        # 向量化填充分区数据 (One-Hot Encoding)
        # 1. 创建节点索引 (0 to batch_size * num_nodes - 1)
        node_indices = torch.arange(batch_size * num_nodes, device=self.device)
        # 2. 创建分区索引 (flattened partition assignments)
        partition_indices = torch.tensor(partition_assignments_np.flatten(), dtype=torch.long, device=self.device)
        # 3. 使用 scatter_ 在指定位置填充 1.0
        batch_features.scatter_(1, partition_indices.unsqueeze(1), 1.0)
        
        # 向量化填充固定特征 (权重和度)
        # Repeat fixed features for each graph in the batch
        batch_features[:, self.num_partitions:] = self.fixed_features_tensor.repeat(batch_size, 1)
        
        return batch_features

    def act(self, graph_data):
        # === 修复：正确解构图数据字典 ===
        state_tensor = graph_data['node_features'].to(self.device)
        edge_index = graph_data['edge_index'].to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # 将特征和边索引都传给模型
            action_values = self.model(state_tensor, edge_index)
        self.model.train()

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
        else:
            action = action_values.argmax().item()
        
        return action

    def replay(self):
        """从经验回放中学习 - 高度优化的向量化批处理版本"""
        if self.current_memory_size < self.batch_size:
            return 0.0
        
        # 随机采样批量索引
        indices = np.random.choice(self.current_memory_size, self.batch_size, replace=False)
        
        # 获取批量数据 (NumPy arrays)
        batch_partition_np = self.memory['partition_assignments'][indices]
        batch_next_partition_np = self.memory['next_partition_assignments'][indices]
        # Transfer other data to GPU
        batch_actions = torch.tensor(self.memory['actions'][indices], dtype=torch.long, device=self.device)
        batch_rewards = torch.tensor(self.memory['rewards'][indices], dtype=torch.float32, device=self.device)
        batch_dones = torch.tensor(self.memory['dones'][indices], dtype=torch.float32, device=self.device)
        
        # ========= 优化: 批量特征转换 (on GPU) =========
        batch_x = self._partition_to_features_batch(batch_partition_np)
        batch_next_x = self._partition_to_features_batch(batch_next_partition_np)
        
        # ========= 优化: 高效构建批处理边索引 =========
        # Use caching mechanism
        if not hasattr(self, '_cached_batch_edge_indices') or self._cached_batch_size != self.batch_size:
            edge_indices_list = []
            for i in range(self.batch_size):
                edge_indices_list.append(self.edge_index + i * self.num_nodes) # self.edge_index is on device
            self._cached_batch_edge_indices = torch.cat(edge_indices_list, dim=1)
            self._cached_batch_size = self.batch_size
        batch_edge_index = self._cached_batch_edge_indices
        
        # ========= 计算当前Q值 (on GPU) =========
        # Shape: [batch_size * num_nodes, num_partitions]
        current_q_values_all = self.model(batch_x, batch_edge_index)
        
        # ========= 优化: 向量化Q值选择 =========
        # 从 batch_actions 解码 node_id 和 partition_id
        # Note: actions are global (0 to num_nodes * num_partitions - 1)
        # We need to map the action's node_id to the global node index in the batch
        batch_node_ids_local = batch_actions // self.num_partitions # Node ID within each graph (0 to num_nodes-1)
        batch_partition_ids = batch_actions % self.num_partitions # Target partition ID (0 to num_partitions-1)
        
        # 计算全局节点索引 (index into the flattened batch_x / current_q_values_all)
        batch_indices_tensor = torch.arange(self.batch_size, device=self.device)
        global_node_indices = batch_indices_tensor * self.num_nodes + batch_node_ids_local
        
        # 使用 gather 或直接索引提取选定的Q值
        # Shape: [batch_size]
        selected_q_values = current_q_values_all[global_node_indices, batch_partition_ids]
        
        # ========= 核心修改: 实现Double DQN逻辑以修复最大化偏差 =========
        with torch.no_grad():
            # 1. 动作选择：使用在线网络(self.model)为下一状态选择最佳动作
            #    这能反映出当前策略认为的最佳选择
            next_q_from_online_model = self.model(batch_next_x, batch_edge_index)
            #    将Q值展平，为每个图找到最佳动作的索引
            best_next_actions = next_q_from_online_model.view(self.batch_size, -1).argmax(dim=1)

            # 2. 价值评估：使用目标网络(self.target_model)来评估被选中动作的价值
            #    这提供了一个更稳定的价值估计，避免了自我膨胀
            next_q_from_target_model = self.target_model(batch_next_x, batch_edge_index)
            next_q_from_target_model_flat = next_q_from_target_model.view(self.batch_size, -1)

            #    使用在线网络选出的 best_next_actions 来从目标网络中提取Q值
            #    .gather() 是实现这一目标的高效方法
            q_value_of_best_action = next_q_from_target_model_flat.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            # 3. 计算最终的目标Q值
            #    如果一个状态是终止状态 (done=1)，则其未来价值为0
            target_q_values = batch_rewards + (1 - batch_dones) * self.gamma * q_value_of_best_action
        
        # 计算损失
        loss = F.mse_loss(selected_q_values, target_q_values)
        
        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 探索率衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新训练计数器并检查是否需要更新目标网络
        self.train_count += 1
        if self.train_count % self.target_update_freq == 0:
            self.update_target_model()
        
        return loss.item()

    def update_target_model(self):
        """更新目标网络参数"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        index = self.memory_counter % self.memory_capacity
        
        # 提取分区分配 (NumPy array of integers)
        partition_assignment = np.argmax(state[:, :self.num_partitions], axis=1).astype(np.int32)
        next_partition_assignment = np.argmax(next_state[:, :self.num_partitions], axis=1).astype(np.int32)
        
        # 存储经验
        self.memory['partition_assignments'][index] = partition_assignment
        self.memory['actions'][index] = action
        self.memory['rewards'][index] = reward
        self.memory['next_partition_assignments'][index] = next_partition_assignment
        self.memory['dones'][index] = float(done)
        
        self.memory_counter += 1
        self.current_memory_size = min(self.memory_counter, self.memory_capacity)

    def save_model(self, filepath):
        """保存模型到文件"""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """从文件加载模型"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_model.load_state_dict(torch.load(filepath, map_location=self.device))