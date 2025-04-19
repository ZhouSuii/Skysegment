# agent_ppo_gnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


# GNN-PPO策略网络
class GNNPPOPolicy(nn.Module):
    def __init__(self, node_features, hidden_dim, output_dim, num_partitions):
        super(GNNPPOPolicy, self).__init__()
        self.num_partitions = num_partitions

        # GNN编码层
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Actor网络（输出动作概率）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

        # Critic网络（估计状态价值）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 节点嵌入
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # 对每个节点输出动作概率和状态价值
        action_probs = self.actor(x)
        values = self.critic(x)

        return action_probs, values

    def act(self, data):
        """根据当前状态选择动作"""
        action_probs, _ = self.forward(data)
        action_probs_flat = action_probs.view(-1)

        # 创建动作分布并采样
        dist = Categorical(action_probs_flat)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.item(), action_log_prob

    def evaluate(self, data, action):
        """评估动作的价值和概率"""
        action_probs, values = self.forward(data)
        action_probs_flat = action_probs.view(-1)

        dist = Categorical(action_probs_flat)
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy()

        return action_log_probs, values.view(-1), entropy


# GNN-PPO智能体
class GNNPPOAgent:
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

        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.batch_size = config.get('batch_size', 32)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.hidden_dim = config.get('hidden_dim', 128)

        # 初始化策略网络
        self.policy = GNNPPOPolicy(self.node_features, self.hidden_dim,
                                   num_partitions, num_partitions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # 用于存储轨迹
        self.states = []
        self.state_datas = []  # 存储PyG数据对象
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

        # 构建固定的图结构
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

        return data

    def act(self, state):
        """根据当前状态选择动作"""
        # 将状态转换为PyG数据
        state_data = self._state_to_pyg_data(state)

        with torch.no_grad():
            action, log_prob = self.policy.act(state_data)
            _, values = self.policy(state_data)
            value = values.mean().item()  # 使用所有节点值的平均作为状态价值

        # 存储当前轨迹信息
        self.states.append(state)
        self.state_datas.append(state_data)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

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
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        total_loss = 0.0
        update_rounds = 0

        # 进行多轮更新
        for _ in range(self.ppo_epochs):
            # 按批次进行更新
            for idx in range(0, len(self.state_datas), self.batch_size):
                batch_idx = slice(idx, min(idx + self.batch_size, len(self.state_datas)))

                batch_state_datas = Batch.from_data_list(
                    [self.state_datas[i] for i in range(*batch_idx.indices(len(self.state_datas)))])
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # 获取当前策略的动作概率、价值估计和熵
                new_log_probs, values, entropy = self.policy.evaluate(batch_state_datas, batch_actions)

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

        # 获取下一个状态的值估计
        with torch.no_grad():
            if self.dones[-1]:
                next_value = 0
            else:
                next_state_data = self._state_to_pyg_data(self.states[-1])
                _, values = self.policy(next_state_data)
                next_value = values.mean().item()

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
        self.state_datas = []
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