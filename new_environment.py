# GraphPartitionEnvironment class
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import copy
# Import the metrics functions
from metrics import calculate_partition_weights, calculate_weight_variance, \
                    calculate_edge_cut, calculate_normalized_cut, calculate_modularity

# 1. 图划分环境 -- environment：定义了action, reward, state
class GraphPartitionEnvironment(gym.Env):
    # CHANGED: 添加了 gamma 和 potential_weights 参数
    def __init__(self, graph, num_partitions=2, max_steps=200, gamma=0.99, potential_weights=None):
        super(GraphPartitionEnvironment, self).__init__()
        self.original_graph = graph
        self.graph = copy.deepcopy(graph)           # deepcopy：创建新的对象与原对象分离
        self.num_nodes = len(graph.nodes())
        self.num_partitions = num_partitions

        # action space: 选择一个节点并将其分配到一个分区
        self.action_space = spaces.Discrete(self.num_nodes * self.num_partitions)

        # state space
        # 形状 = [self.num_nodes, self.num_partitions + 2]
        self.observation_space = spaces.Box(    # box：连续空间
            low=0, high=1,
            shape=(self.num_nodes, self.num_partitions + 2),
            dtype=np.float32
        )

        # 初始化分区分配
        self.partition_assignment = np.zeros(self.num_nodes, dtype=int)
        self.node_weights = np.array([graph.nodes[i].get('weight', 1) for i in range(self.num_nodes)]) # Default weight to 1

        degrees = np.array([graph.degree[i] for i in range(self.num_nodes)], dtype=float)
        max_degree = max(degrees) if len(degrees) > 0 else 1.0
        self.node_degrees = degrees / max_degree

        self.max_steps = max_steps
        self.current_step = 0

        # PBRS Parameters
        self.gamma = gamma
        # Define default potential weights if not provided
        default_weights = {'variance': 1.0, 'edge_cut': 1.0, 'modularity': 1.0}
        self.potential_weights = potential_weights if potential_weights is not None else default_weights

        # Pre-calculate normalization factors for potential function metrics (optional but recommended)
        self._total_weight_sq = np.sum(self.node_weights)**2 + 1e-6 # For normalizing variance
        self._total_edges = len(self.graph.edges()) + 1e-6 # For normalizing edge cut
        # Modularity is already normalized to [-1, 1]


    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed) # Use gymnasium's reset seed
        else:
             super().reset()

        # 随机初始化分区
        self.partition_assignment = np.random.randint(0, self.num_partitions, self.num_nodes)

        # 重置步数
        self.current_step = 0

        # Return initial state and info
        return self._get_state(), {}


    def _get_state(self):
        """改进状态表示，确保特征有足够区分度"""
        # 状态组成：one-hot 分区、归一化节点度、原始节点权重
        state = np.zeros((self.num_nodes, self.num_partitions + 2), dtype=np.float32)
        
        # One-hot编码分区分配
        for i in range(self.num_nodes):
            state[i, self.partition_assignment[i]] = 1.0
        
        # 添加节点权重（不要过度归一化）
        max_weight = np.max(self.node_weights) if len(self.node_weights) > 0 else 1.0
        if max_weight > 0:
            normalized_weights = self.node_weights / max_weight
        else:
            normalized_weights = np.ones_like(self.node_weights)
        
        # 添加节点度（归一化）
        for i in range(self.num_nodes):
            state[i, self.num_partitions] = self.node_degrees[i]
            state[i, self.num_partitions + 1] = normalized_weights[i]
        return state

    # 新方法：计算给定分区分配状态的势能
    def _calculate_potential(self, partition_assignment):
        """
        Calculates the potential function value for a given partition assignment.
        Φ(s) = -w_var * NormalizedVariance(s) - w_cut * NormalizedEdgeCut(s) + w_mod * Modularity(s)
        Higher potential is better.
        """
        # 确保分区分配索引有效 (防御性编程)
        valid_indices = np.all([(pa >= 0 and pa < self.num_partitions) for pa in partition_assignment])
        if not valid_indices:
            # 若新方法：计算给定分区分配状态的势能
             return -1e9

        # 使用 metrics.py 中的辅助函数计算指标
        variance = calculate_weight_variance(self.graph, partition_assignment, self.num_partitions)
        edge_cut = calculate_edge_cut(self.graph, partition_assignment)
        modularity = calculate_modularity(self.graph, partition_assignment, self.num_partitions)

        # Normalize metrics
        normalized_variance = variance / self._total_weight_sq # Normalize by total weight squared
        normalized_edge_cut = edge_cut / self._total_edges # Normalize by total edges

        # 根据权重计算势能（假设权重为正）
        # 希望低方差、低切割、高模块度 -> 对 方差/切割 使用负号
        potential = -self.potential_weights.get('variance', 1.0) * normalized_variance \
                    -self.potential_weights.get('edge_cut', 1.0) * normalized_edge_cut \
                    +self.potential_weights.get('modularity', 1.0) * modularity
        
        return potential


    def step(self, action):
        # 解码动作：节点索引和目标分区
        node_idx = action // self.num_partitions
        target_partition = action % self.num_partitions

        old_partition_assignment = self.partition_assignment.copy() # Keep old assignment for potential calculation

        # 计算当前状态的势能（执行动作之前）
        potential_current = self._calculate_potential(old_partition_assignment)

        # 尝试应用动作
        self.partition_assignment[node_idx] = target_partition

        # 检查有效性（确保没有分区变空）
        partition_counts = np.bincount(self.partition_assignment, minlength=self.num_partitions)
        valid_partition = np.all(partition_counts > 0)

        done = False # Default done state

        if not valid_partition:
            # 如果不是有效分区，则恢复分配
            self.partition_assignment = old_partition_assignment.copy()
            original_reward = -100.0 # 对无效动作进行重罚

            # 我们只对无效动作使用原始惩罚奖励。
            shaped_reward = original_reward

        else:
            # 如果是有效分区，计算原始奖励和塑形奖励
            original_reward = self._calculate_reward() # This is your old linear combination reward

            # 计算下一个状态的势能（执行有效动作之后）
            potential_next = self._calculate_potential(self.partition_assignment)

            # 使用 PBRS 公式计算塑形奖励
            shaped_reward = original_reward + self.gamma * potential_next - potential_current


        # 更新步数并检查是否达到最大步数
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # 返回下一个状态、塑形奖励、完成标志、截断标志 (False)、信息
        return self._get_state(), shaped_reward, done, False, {}

    # for the PBRS calculation in the step method.
    def _calculate_reward(self):
        # 计算分区权重
        partition_weights = np.zeros(self.num_partitions)
        for i in range(self.num_nodes):
            partition_weights[self.partition_assignment[i]] += self.node_weights[i]

        # 使用最大/最小比率计算平衡奖励（原始方法）
        max_weight = np.max(partition_weights) if len(partition_weights) > 0 else 0
        min_weight = np.min(partition_weights[partition_weights > 0]) if np.any(partition_weights > 0) else 1
        if min_weight <= 0: min_weight = 1 # 避免除零或负数

        balance_reward = -1 * (max_weight / min_weight - 1)

        # 计算边切割奖励（原始方法）
        cut_edges = calculate_edge_cut(self.graph, self.partition_assignment) 

        edge_density = len(self.graph.edges()) / max(1.0, (self.num_nodes * (self.num_nodes - 1) / 2)) # Avoid division by zero for single node graph
        normalized_cut = cut_edges / max(1.0, len(self.graph.edges()))
        edge_cut_reward = -normalized_cut * (1 + edge_density)

        
        original_reward = 2.0 * balance_reward + 1.0 * edge_cut_reward
        return original_reward

    # Render method remains the same
    def render(self):
        """绘制当前图划分状态"""
        plt.figure(figsize=(10, 8))

        # 为每个分区分配不同的颜色
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink']
        node_colors = [colors[self.partition_assignment[i] % len(colors)] for i in range(self.num_nodes)]

        # 获取节点位置
        pos = nx.spring_layout(self.graph, seed=42)

        # 绘制节点和边
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_edges(self.graph, pos)

        # 显示节点权重
        node_labels = {i: f"{i}:{self.node_weights[i]}" for i in range(self.num_nodes)}
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels)

        # 计算每个分区的总权重
        partition_weights = np.zeros(self.num_partitions)
        for i in range(self.num_nodes):
            partition_weights[self.partition_assignment[i]] += self.node_weights[i]

        # 添加标题显示分区权重
        plt.title(f"图划分 - 分区权重: {partition_weights}")
        plt.axis('off')
        plt.show()