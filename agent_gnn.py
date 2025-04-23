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
        
        # 创建节点到索引的映射
        self._create_batch_mapping()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
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
        
        # 提取并归一化节点权重
        self.node_weights = np.array([self.graph.nodes[i].get('weight', 1.0)
                                     for i in range(self.num_nodes)], dtype=np.float32)
        max_weight = np.max(self.node_weights) if len(self.node_weights) > 0 else 1.0
        self.node_weights = self.node_weights / (max_weight + 1e-8)  # 避免除零
        
        # 提取并归一化节点度
        degrees = np.array([self.graph.degree[i]
                           for i in range(self.num_nodes)], dtype=np.float32)
        max_degree = np.max(degrees) if len(degrees) > 0 else 1.0
        self.node_degrees = degrees / (max_degree + 1e-8)  # 避免除零
    
    def _create_batch_mapping(self):
        """创建批处理索引映射"""
        self.node_offsets = np.zeros(self.batch_size + 1, dtype=np.int32)
        for i in range(1, self.batch_size + 1):
            self.node_offsets[i] = self.node_offsets[i-1] + self.num_nodes
    
    def _partition_to_features(self, partition_assignment):
        """将分区分配转换为节点特征矩阵"""
        # 创建特征矩阵: [分区one-hot, 归一化权重, 归一化度]
        x = np.zeros((self.num_nodes, self.node_features), dtype=np.float32)
        
        # 高效设置one-hot编码
        x[np.arange(self.num_nodes), partition_assignment] = 1.0
        
        # 添加权重和度特征
        x[:, self.num_partitions] = self.node_weights
        x[:, self.num_partitions + 1] = self.node_degrees
        
        return x
    
    def _state_to_pyg_data(self, partition_assignment):
        """将分区分配转换为PyG Data对象"""
        # 将分区分配转换为节点特征
        x = self._partition_to_features(partition_assignment)
    
    # 直接返回特征张量
        return torch.FloatTensor(x).to(self.device)
    
    def update_target_model(self):
        """更新目标网络参数"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        index = self.memory_counter % self.memory_capacity
        
        # 提取分区分配
        partition_assignment = np.argmax(state[:, :self.num_partitions], axis=1)
        next_partition_assignment = np.argmax(next_state[:, :self.num_partitions], axis=1)
        
        # 存储经验
        self.memory['partition_assignments'][index] = partition_assignment
        self.memory['actions'][index] = action
        self.memory['rewards'][index] = reward
        self.memory['next_partition_assignments'][index] = next_partition_assignment
        self.memory['dones'][index] = float(done)
        
        self.memory_counter += 1
        self.current_memory_size = min(self.memory_counter, self.memory_capacity)
    
    def act(self, state):
        """根据当前状态选择动作"""
        # 探索：随机选择动作
        if np.random.rand() <= self.epsilon:
            # 随机选择节点和目标分区
            node_id = random.randrange(self.num_nodes)
            target_partition = random.randrange(self.num_partitions)
            action = node_id * self.num_partitions + target_partition
            return action
        
        # 利用：使用模型预测最佳动作
        partition_assignment = np.argmax(state[:, :-1], axis=1)
        x_tensor = self._state_to_pyg_data(partition_assignment)
        
        self.model.eval()  # 设置为评估模式
        with torch.no_grad():
            # 获取每个节点对每个分区的Q值
            q_values = self.model(x_tensor, self.edge_index)
            
            # 展平为一维数组以找出最大Q值对应的动作
            q_values_flat = q_values.reshape(-1)
            action = torch.argmax(q_values_flat).item()
        
        self.model.train()  # 恢复训练模式
        return action
    
    def replay(self):
        """从经验回放中学习"""
        if self.current_memory_size < self.batch_size:
            return 0.0
        
        # 随机采样批量索引
        indices = np.random.choice(self.current_memory_size, self.batch_size, replace=False)
        
        # 获取批量数据
        batch_partition = self.memory['partition_assignments'][indices]
        batch_next_partition = self.memory['next_partition_assignments'][indices]
        batch_actions = torch.tensor(self.memory['actions'][indices], dtype=torch.long).to(self.device)
        batch_rewards = torch.tensor(self.memory['rewards'][indices], dtype=torch.float32).to(self.device)
        batch_dones = torch.tensor(self.memory['dones'][indices], dtype=torch.float32).to(self.device)
        
       # 创建批量输入
        batch_xs = []
        batch_next_xs = []
        batch_indices = []
        for i in range(self.batch_size):
            # 获取节点特征
            x_tensor = self._state_to_pyg_data(batch_partition[i])
            next_x_tensor = self._state_to_pyg_data(batch_next_partition[i])
            
            batch_xs.append(x_tensor)
            batch_next_xs.append(next_x_tensor)
            
            # 为批次索引添加样本ID
            batch_indices.extend([i] * self.num_nodes)

        # 连接为大批次
        batch_x = torch.cat(batch_xs, dim=0)
        batch_next_x = torch.cat(batch_next_xs, dim=0)
        batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=self.device)

        # 创建扩展的边索引
        edge_indices = []
        for i in range(self.batch_size):
            # 为每个样本创建边索引
            offset = i * self.num_nodes
            # 克隆边索引并添加偏移
            batch_edge_index = self.edge_index.clone()
            batch_edge_index = batch_edge_index + offset
            edge_indices.append(batch_edge_index)

        # 连接所有边索引
        batch_edge_index = torch.cat(edge_indices, dim=1)

        # 计算当前Q值
        current_q_values = self.model(batch_x, batch_edge_index)
        
        # 从batch_actions解码node_id和partition_id
        node_ids = batch_actions // self.num_partitions
        partition_ids = batch_actions % self.num_partitions
        
        selected_q_values = []
        for i in range(self.batch_size):
            # 计算全局节点索引
            global_node_idx = i * self.num_nodes + node_ids[i]
            # 获取该节点对选定分区的Q值
            if global_node_idx < batch_x.size(0):  # 检查索引是否在范围内
                selected_q_values.append(current_q_values[global_node_idx, partition_ids[i]])
                
        if not selected_q_values:  # 如果列表为空
            print("警告: 没有有效的Q值可用于训练")
            return 0.0
            
        # 转换为张量
        selected_q_values = torch.stack(selected_q_values)
        
        # --- 使用向量化计算目标Q值 ---
        with torch.no_grad():
            # 计算下一状态的Q值
            next_q_values = self.target_model(batch_next_x, batch_edge_index)
            
            # 找出每个图中最大的Q值
            max_q_values = []
            for i in range(self.batch_size):
                # 计算当前批次的节点范围
                start_idx = i * self.num_nodes
                end_idx = (i + 1) * self.num_nodes
                
                # 获取当前样本中所有节点的Q值
                batch_q_values = next_q_values[start_idx:end_idx]
                
                # 找出每个节点最大的Q值
                node_max_q = batch_q_values.max(dim=1)[0]
                
                # 然后找出该样本所有节点中的最大值
                max_q_values.append(node_max_q.max())
            
            # 转换为张量
            max_q_values = torch.stack(max_q_values)
            
            # 计算目标Q值
            target_q_values = batch_rewards + (1 - batch_dones) * self.gamma * max_q_values
        
        # 计算损失
        loss = F.mse_loss(selected_q_values, target_q_values)
        
        # 调试信息 - 检查loss是否为0或NaN
        if loss.item() == 0 or torch.isnan(loss):
            print(f"警告: Loss值异常: {loss.item()}")
            print(f"选定Q值: {selected_q_values}")
            print(f"目标Q值: {target_q_values}")
            print(f"奖励范围: [{batch_rewards.min().item()}, {batch_rewards.max().item()}]")
        
        # 梯度更新
        self.optimizer.zero_grad()
        loss.backward()
        # 添加梯度剪裁以提高稳定性
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

    def save_model(self, filepath):
        """保存模型到文件"""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """从文件加载模型"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_model.load_state_dict(torch.load(filepath, map_location=self.device))