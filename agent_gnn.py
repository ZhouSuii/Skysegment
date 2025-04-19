# gnn agent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from torch_geometric.nn import GCNConv, global_mean_pool
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

        # 如果是批处理数据，则获取batch_size
        # 否则默认为1（单个图）
        if hasattr(data, 'num_graphs'):
            batch_size = data.num_graphs
        else:
            batch_size = 1

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
        self.memory_size = config.get('memory_size', 2000)

        # 初始化经验回放
        self.memory = deque(maxlen=self.memory_size)

        # 初始化GNN模型
        hidden_dim = config.get('hidden_dim', 128)
        self.model = GNNDQN(self.node_features, hidden_dim, num_partitions, num_partitions)
        self.target_model = GNNDQN(self.node_features, hidden_dim, num_partitions, num_partitions)
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

        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 保存节点权重和度
        self.node_weights = np.array([self.graph.nodes[i]['weight']
                                      for i in range(self.num_nodes)], dtype=float)
        max_weight = max(self.node_weights) if len(self.node_weights) > 0 else 1.0
        self.node_weights = self.node_weights / max_weight

        degrees = np.array([self.graph.degree[i]
                            for i in range(self.num_nodes)], dtype=float)
        max_degree = max(degrees) if len(degrees) > 0 else 1.0
        self.node_degrees = degrees / max_degree

    def _state_to_pyg_data(self, state):
        """将环境状态转换为PyG数据对象"""
        # 状态形状: [num_nodes, num_partitions + 1]
        # 最后一列是节点度

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

        # 为单个图模拟batch
        data.batch = torch.zeros(self.num_nodes, dtype=torch.long)

        return data

    def update_target_model(self):
        """更新目标网络的参数"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

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
        """从经验回放中学习"""
        if len(self.memory) < batch_size:
            return 0.0  # 返回默认loss值

        total_loss = 0.0
        # 从记忆中随机采样
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            # 将状态转换为PyG数据
            state_data = self._state_to_pyg_data(state)
            next_state_data = self._state_to_pyg_data(next_state)

            # 计算目标 Q 值
            if done:
                target = reward
            else:
                with torch.no_grad():
                    # 使用PyG数据对象而非张量
                    next_q_values = self.target_model(next_state_data)
                    target = reward + self.gamma * torch.max(next_q_values).item()

            # 获取当前 Q 值预测
            current_q = self.model(state_data)

            # 找出选中的动作对应的Q值
            # 将action展平为节点索引和分区索引
            node_idx = action // self.num_partitions
            partition_idx = action % self.num_partitions

            # 获取选中节点和分区的Q值
            current_q_value = current_q[node_idx, partition_idx]

            # 执行优化步骤
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(current_q_value, torch.tensor([target]))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # 探索率衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 更新训练计数器并检查是否需要更新目标网络
        self.train_count += 1
        if self.train_count % self.target_update_freq == 0:
            self.update_target_model()

        return total_loss / batch_size  # 返回平均损失

    def save_model(self, filepath):
        """保存模型到文件"""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """从文件加载模型"""
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(torch.load(filepath))