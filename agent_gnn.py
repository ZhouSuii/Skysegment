# gnn agent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch_geometric.nn import GCNConv
import networkx as nx
from torch_geometric.data import Data, Batch


# GNN-DQN网络模型
class GNNDQN(nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim, num_partitions):
        super(GNNDQN, self).__init__()
        self.num_partitions = num_partitions

        # GNN层
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Q值预测层
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 应用GNN层
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # 对每个节点，为每个可能的分区预测Q值
        q_values = self.q_net(x)

        return q_values


# GNN-DQN智能体
class GNNDQNAgent:
    def __init__(self, graph, num_partitions, config=None):
        self.graph = graph
        self.num_partitions = num_partitions
        self.num_nodes = len(graph.nodes())

        # 节点特征维度: 当前分区分配(one-hot) + 节点权重 + 节点度
        self.node_features = num_partitions + 2
        self.action_size = self.num_nodes * num_partitions

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
        
        # 检查GPU可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GNN-DQN使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")

        # 初始化经验回放缓冲区
        self.memory_counter = 0
        self.memory = {
            'states': [],  # 存储PyG数据对象
            'actions': np.zeros(self.memory_capacity, dtype=np.int64),
            'rewards': np.zeros(self.memory_capacity, dtype=np.float32),
            'next_states': [],  # 存储PyG数据对象
            'dones': np.zeros(self.memory_capacity, dtype=np.float32)
        }

        # 初始化GNN模型并移动到GPU
        hidden_dim = config.get('hidden_dim', 128)
        self.model = GNNDQN(self.node_features, hidden_dim, num_partitions, num_partitions).to(self.device)
        self.target_model = GNNDQN(self.node_features, hidden_dim, num_partitions, num_partitions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

        # 训练计数器
        self.train_count = 0

        # 创建固定的图结构
        self._build_graph_structure()

    def _build_graph_structure(self):
        """构建PyG的图数据结构"""
        # 创建边索引
        edge_index = []
        for u, v in self.graph.edges():
            edge_index.append([u, v])
            edge_index.append([v, u])  # 添加反向边

        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)

        # 保存节点权重和度
        self.node_weights = np.array([self.graph.nodes[i]['weight']
                                      for i in range(self.num_nodes)], dtype=np.float32)
        max_weight = max(self.node_weights) if len(self.node_weights) > 0 else 1.0
        self.node_weights = self.node_weights / max_weight

        degrees = np.array([self.graph.degree[i]
                            for i in range(self.num_nodes)], dtype=np.float32)
        max_degree = max(degrees) if len(degrees) > 0 else 1.0
        self.node_degrees = degrees / max_degree

    def _state_to_pyg_data(self, state):
        """将环境状态转换为PyG数据对象"""
        # 创建节点特征: [分区one-hot, 归一化权重, 归一化度]
        x = np.zeros((self.num_nodes, self.node_features), dtype=np.float32)

        # 填充分区one-hot部分
        for i in range(self.num_nodes):
            partition = np.argmax(state[i, :-1])  # 找到节点所属的分区
            x[i, partition] = 1.0

            # 添加节点权重和度
            x[i, self.num_partitions] = self.node_weights[i]
            x[i, self.num_partitions + 1] = self.node_degrees[i]

        # 创建PyG数据对象
        x = torch.FloatTensor(x)
        data = Data(x=x, edge_index=self.edge_index)
        
        # 将数据移至GPU设备
        data = data.to(self.device)

        return data

    def update_target_model(self):
        """更新目标网络的参数"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        index = self.memory_counter % self.memory_capacity
        
        # 转换为PyG数据对象并存储
        state_data = self._state_to_pyg_data(state)
        next_state_data = self._state_to_pyg_data(next_state)
        
        # 如果列表长度小于当前索引+1，则添加新元素
        if len(self.memory['states']) <= index:
            self.memory['states'].append(state_data)
            self.memory['next_states'].append(next_state_data)
        else:
            # 否则替换已有元素
            self.memory['states'][index] = state_data
            self.memory['next_states'][index] = next_state_data
            
        # 存储其他数据
        self.memory['actions'][index] = action
        self.memory['rewards'][index] = reward
        self.memory['dones'][index] = float(done)
        
        self.memory_counter += 1

    def act(self, state):
        """根据当前状态选择动作"""
        # 探索：随机选择动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # 利用：使用模型预测最佳动作
        state_data = self._state_to_pyg_data(state)
        with torch.no_grad():
            q_values = self.model(state_data)

        # 重塑Q值为节点和分区
        q_values = q_values.view(-1)  # 展平为[num_nodes * num_partitions]
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        """从经验回放中学习，使用真正的批处理"""
        # 确保有足够的样本
        memory_size = min(self.memory_counter, self.memory_capacity)
        if memory_size < batch_size:
            return 0.0

        # 随机采样批量索引
        indices = np.random.choice(memory_size, batch_size, replace=False)
        
        # 从内存中获取批处理数据
        batch_states = [self.memory['states'][i] for i in indices]
        batch_next_states = [self.memory['next_states'][i] for i in indices]
        
        # 使用Batch类创建批处理图数据
        batch_state_data = Batch.from_data_list(batch_states).to(self.device)
        batch_next_state_data = Batch.from_data_list(batch_next_states).to(self.device)
        
        # 将其他数据转移到GPU
        batch_actions = torch.tensor(self.memory['actions'][indices], dtype=torch.long).to(self.device)
        batch_rewards = torch.tensor(self.memory['rewards'][indices], dtype=torch.float32).to(self.device)
        batch_dones = torch.tensor(self.memory['dones'][indices], dtype=torch.float32).to(self.device)

        # 获取当前Q值
        current_q_values = self.model(batch_state_data)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_model(batch_next_state_data)
            
            # 对每个节点获取最大Q值，并根据批处理索引分组
            next_max_q = torch.zeros(batch_size, dtype=torch.float32).to(self.device)
            
            # 从索引中获取每个节点的批处理id
            batch_idx = batch_next_state_data.batch
            
            # 获取每个图中每个节点的最大Q值
            for i in range(batch_size):
                # 找到属于当前批次索引的节点
                mask = (batch_idx == i)
                # 获取这些节点的Q值并找出最大值
                next_max_q[i] = next_q_values[mask].max()
                
            # 计算目标Q值
            target_q = batch_rewards + (1 - batch_dones) * self.gamma * next_max_q
        
        # 准备当前Q值用于损失计算
        batch_indices = []
        node_indices = []
        q_values = []
        
        # 计算actions对应的节点索引和分区
        node_idx_list = batch_actions // self.num_partitions
        partition_idx_list = batch_actions % self.num_partitions
        
        # 处理批量的每个示例
        for batch_idx in range(batch_size):
            # 获取当前示例中的节点索引
            node_idx = node_idx_list[batch_idx]
            partition_idx = partition_idx_list[batch_idx]
            
            # 找出该节点在批处理图中的索引
            actual_node_idx = (batch_state_data.batch == batch_idx).nonzero(as_tuple=True)[0][node_idx]
            
            # 添加到集合中
            q_values.append(current_q_values[actual_node_idx, partition_idx])
        
        # 将q_values转换为张量
        q_values = torch.stack(q_values)
        
        # 计算损失并优化
        loss = nn.MSELoss()(q_values, target_q)
        
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
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_model.load_state_dict(torch.load(filepath, map_location=self.device))