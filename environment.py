# GraphPartitionEnvironment class
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import copy

# 1. 图划分环境 -- environment：定义了action, reward, state
class GraphPartitionEnvironment(gym.Env):
    def __init__(self, graph, num_partitions=2, max_steps=200):  # CHANGED: 增加 max_steps 参数
        super(GraphPartitionEnvironment, self).__init__()
        self.original_graph = graph
        self.graph = copy.deepcopy(graph)           # deepcopy：创建新的对象与原对象分离
        self.num_nodes = len(graph.nodes())
        self.num_partitions = num_partitions

        # action space: 选择一个节点并将其分配到一个分区
        # 分离动作选择：先选节点，再选分区
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.num_nodes),  # 选择节点
            spaces.Discrete(self.num_partitions)  # 选择目标分区
        ))

        # state space
        # CHANGED: 状态增加节点的度信息，因此多出一列，用于存储归一化后的节点度
        # 形状 = [self.num_nodes, self.num_partitions + 1]
        self.observation_space = spaces.Box(    # box：连续空间
            low=0, high=1,
            shape=(self.num_nodes, self.num_partitions + 1),
            dtype=np.float32
        )

        # 初始化分区分配
        self.partition_assignment = np.zeros(self.num_nodes, dtype=int)
        self.node_weights = np.array([graph.nodes[i]['weight'] for i in range(self.num_nodes)])

        # CHANGED: 保存节点度，并对其进行归一化
        degrees = np.array([graph.degree[i] for i in range(self.num_nodes)], dtype=float)
        max_degree = max(degrees) if len(degrees) > 0 else 1.0
        self.node_degrees = degrees / max_degree

        # CHANGED: 增加记录当前步数的属性
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # 随机初始化分区
        self.partition_assignment = np.random.randint(0, self.num_partitions, self.num_nodes)

        # CHANGED: 重置步数
        self.current_step = 0

        return self._get_state(), {}

    def _get_state(self):
        # CHANGED: 状态由分区分配的 one-hot 和节点度组合而成
        state = np.zeros((self.num_nodes, self.num_partitions + 1), dtype=np.float32)
        for i in range(self.num_nodes):
            state[i, self.partition_assignment[i]] = 1.0
            # 将节点度放在最后一列
            state[i, -1] = self.node_degrees[i]
        return state

    def step(self, action):
        # 解码动作: 节点索引和目标分区
        node_idx = action // self.num_partitions
        target_partition = action % self.num_partitions

        old_partition = self.partition_assignment[node_idx]
        self.partition_assignment[node_idx] = target_partition

        # 检查是否所有分区至少有一个节点
        partition_counts = np.bincount(self.partition_assignment, minlength=self.num_partitions)
        valid_partition = np.all(partition_counts > 0)

        if not valid_partition:
            # 如果不是有效分区，恢复操作
            self.partition_assignment[node_idx] = old_partition
            reward = -1.0
            done = False
        else:
            reward = self._calculate_reward()
            done = False  # 可以在此处根据需求改进结束条件

        # CHANGED: 步数加一，并检查是否超出最大步数
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_state(), reward, done, False, {}

# changed2
    def _calculate_reward(self):
        # 计算分区权重平衡度
        partition_weights = np.zeros(self.num_partitions)
        for i in range(self.num_nodes):
            partition_weights[self.partition_assignment[i]] += self.node_weights[i]

        # 使用最大权重与最小权重的比值代替方差 -- 方差受到权重绝对值大小的影响，比值相对平衡
        max_weight = max(partition_weights)
        min_weight = min(partition_weights) if min(partition_weights) > 0 else 1
        balance_reward = -1 * (max_weight / min_weight - 1)

        # 切割边权重惩罚 -- 图密度作为加权因子
        cut_edges = 0
        for u, v in self.graph.edges():
            if self.partition_assignment[u] != self.partition_assignment[v]:
                cut_edges += 1

        edge_density = len(self.graph.edges()) / (self.num_nodes * (self.num_nodes - 1) / 2)
        normalized_cut = cut_edges / max(1, len(self.graph.edges()))
        edge_cut_reward = -normalized_cut * (1 + edge_density)

        # 组合奖励 -- 移除连通性奖励 -- 切割边间接鼓励连通性
        total_reward = 2.0 * balance_reward + 1.0 * edge_cut_reward
        return total_reward

    # 绘图
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