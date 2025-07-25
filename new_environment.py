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
        self._total_weight_sq = np.sum(self.node_weights)**2 + 1e-6
        self._total_edges = len(self.graph.edges()) + 1e-6

        # === 新增：边索引缓存优化 ===
        self._cached_edge_index = None
        self._cached_edge_index_torch = None
        self._edge_index_initialized = False
        self._initialize_edge_index_cache()

    def _initialize_edge_index_cache(self):
        """=== 新增：初始化边索引缓存 ==="""
        try:
            import torch
            
            # 构建边索引 (只需计算一次，因为图拓扑不变)
            edges = list(self.graph.edges())
            if len(edges) > 0:
                # 创建无向图的边索引（每条边两个方向）
                edge_list = []
                for u, v in edges:
                    edge_list.append([u, v])
                    edge_list.append([v, u])  # 反向边
                self._cached_edge_index = np.array(edge_list, dtype=np.int64).T  # [2, num_edges]
                self._cached_edge_index_torch = torch.tensor(self._cached_edge_index, dtype=torch.long)
            else:
                # 如果没有边，创建空的边索引
                self._cached_edge_index = np.zeros((2, 0), dtype=np.int64)
                self._cached_edge_index_torch = torch.zeros((2, 0), dtype=torch.long)
            
            self._edge_index_initialized = True
            print(f"边索引缓存初始化完成 - 边数: {self._cached_edge_index.shape[1]}")
            
        except ImportError:
            # 如果torch不可用，在需要时动态计算
            print("PyTorch不可用，将在需要时动态计算边索引")
            self._edge_index_initialized = False

    def reset(self, seed=None, state_format='matrix'):
        """
        重置环境
        Args:
            seed: 随机种子
            state_format: 'matrix' 用于传统方法, 'graph' 用于GNN方法
        """
        if seed is not None:
            super().reset(seed=seed) # Use gymnasium's reset seed
        else:
             super().reset()

        # 随机初始化分区
        self.partition_assignment = np.random.randint(0, self.num_partitions, self.num_nodes)

        # 重置步数
        self.current_step = 0

        # Return initial state and info
        return self.get_state(format=state_format), {}


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

    # === 新增：为GNN返回图结构数据的方法 ===
    def _get_graph_state(self):
        """
        === 优化：为GNN智能体返回图结构数据格式，使用边索引缓存 ===
        返回包含节点特征、边索引和分区信息的字典
        """
        import torch
        
        # 1. 构建节点特征矩阵
        node_features = np.zeros((self.num_nodes, self.num_partitions + 2), dtype=np.float32)
        
        # One-hot编码当前分区分配
        for i in range(self.num_nodes):
            node_features[i, self.partition_assignment[i]] = 1.0
        
        # 添加节点度（归一化）
        node_features[:, self.num_partitions] = self.node_degrees
        
        # 添加节点权重（归一化）
        max_weight = np.max(self.node_weights) if len(self.node_weights) > 0 else 1.0
        if max_weight > 0:
            normalized_weights = self.node_weights / max_weight
        else:
            normalized_weights = np.ones_like(self.node_weights)
        node_features[:, self.num_partitions + 1] = normalized_weights
        
        # 2. === 优化：使用缓存的边索引 ===
        if self._edge_index_initialized and self._cached_edge_index_torch is not None:
            # 使用预计算的边索引缓存
            edge_index = self._cached_edge_index_torch.clone()  # 克隆以避免意外修改
        else:
            # 回退到动态计算（仅在缓存失败时）
            edges = list(self.graph.edges())
            if len(edges) > 0:
                edge_list = []
                for u, v in edges:
                    edge_list.append([u, v])
                    edge_list.append([v, u])  # 反向边
                edge_index = torch.tensor(np.array(edge_list, dtype=np.int64).T, dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # 3. 当前分区分配
        current_partition = self.partition_assignment.copy()
        
        # 返回图结构数据
        graph_data = {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': edge_index,
            'current_partition': torch.tensor(current_partition, dtype=torch.long)
        }
        
        return graph_data

    # === 修改：支持两种状态格式的统一接口 ===
    def get_state(self, format='matrix'):
        """
        统一的状态获取接口
        Args:
            format: 'matrix' 用于传统方法, 'graph' 用于GNN方法
        """
        if format == 'graph':
            return self._get_graph_state()
        else:
            return self._get_state()

    # 新方法：计算给定分区分配状态的势能
    def _calculate_potential(self, partition_assignment):
        """
        分层势函数设计：优先满足约束条件，然后优化目标
        
        Φ(s) = Φ_constraint(s) + Φ_objective(s)
        
        其中：
        - Φ_constraint(s): 确保合法分区分配（负值惩罚违规）
        - Φ_objective(s): 在满足约束的基础上优化质量指标
        """
        # 确保分区分配索引有效（防御性编程）
        valid_indices = np.all([(pa >= 0 and pa < self.num_partitions) for pa in partition_assignment])
        if not valid_indices:
            return -1e9

        # 第1层：约束条件检查
        constraint_potential = self._calculate_constraint_potential(partition_assignment)
        
        # 第2层：目标优化（只有在约束满足时才起作用）
        objective_potential = self._calculate_objective_potential(partition_assignment)
        
        # 分层权重：约束条件权重更高
        total_potential = 3.0 * constraint_potential + 1.0 * objective_potential
        
        return total_potential
    
    def _calculate_constraint_potential(self, partition_assignment):
        """计算约束条件势函数：确保分区平衡和连通性"""
        constraint_score = 0.0
        
        # 约束1：分区平衡性 - 每个分区至少要有一个节点
        partition_counts = np.bincount(partition_assignment, minlength=self.num_partitions)
        empty_partitions = np.sum(partition_counts == 0)
        
        if empty_partitions > 0:
            # 严重惩罚空分区
            constraint_score -= 100.0 * empty_partitions
        
        # 约束2：分区大小不应过于不均衡
        mean_partition_size = self.num_nodes / self.num_partitions
        max_imbalance = np.max(np.abs(partition_counts - mean_partition_size)) / mean_partition_size
        
        if max_imbalance > 0.5:  # 允许50%的不平衡
            constraint_score -= 10.0 * (max_imbalance - 0.5)
        
        # === 新增约束3：确保每个分区内部连通 ===
        disconnected_partitions = self._count_disconnected_partitions(partition_assignment)
        if disconnected_partitions > 0:
            # 对每个非连通分区进行惩罚
            # 惩罚力度设置得比空分区稍小，但仍然显著
            constraint_score -= 50.0 * disconnected_partitions
        
        return constraint_score
    
    def _count_disconnected_partitions(self, partition_assignment):
        """计算非连通分区的数量"""
        disconnected_count = 0
        
        # 为每个分区构建子图并检查连通性
        for partition_id in range(self.num_partitions):
            # 找到属于当前分区的所有节点
            partition_nodes = [i for i in range(self.num_nodes) 
                             if partition_assignment[i] == partition_id]
            
            if len(partition_nodes) <= 1:
                # 单节点或空分区自然是连通的（或无意义）
                continue
            
            # 构建当前分区的子图
            subgraph = self.graph.subgraph(partition_nodes)
            
            # 检查子图是否连通
            if not nx.is_connected(subgraph):
                disconnected_count += 1
        
        return disconnected_count
    
    def _calculate_objective_potential(self, partition_assignment):
        """计算目标优化势函数：权重方差、边切割、模块度"""
        # 使用 metrics.py 中的辅助函数计算指标
        variance = calculate_weight_variance(self.graph, partition_assignment, self.num_partitions)
        edge_cut = calculate_edge_cut(self.graph, partition_assignment)
        modularity = calculate_modularity(self.graph, partition_assignment, self.num_partitions)

        # 渐进式归一化：避免极端值主导
        normalized_variance = np.tanh(variance / self._total_weight_sq)  # 使用tanh限制范围
        normalized_edge_cut = np.tanh(edge_cut / self._total_edges)      # 使用tanh限制范围
        normalized_modularity = np.tanh(modularity)                      # 模块度已在[-1,1]范围
        
        # 目标函数：希望低方差、低切割、高模块度
        objective_potential = (-self.potential_weights.get('variance', 1.0) * normalized_variance 
                             - self.potential_weights.get('edge_cut', 1.0) * normalized_edge_cut 
                             + self.potential_weights.get('modularity', 1.0) * normalized_modularity)
        
        return objective_potential


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
        # 注意：这里仍然返回矩阵格式，GNN智能体需要额外调用get_state('graph')
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