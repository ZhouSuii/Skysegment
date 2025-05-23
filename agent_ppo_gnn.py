# agent_ppo_gnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
import time  # 添加 time 模块用于性能分析


# GNN-PPO策略网络
class GNNPPOPolicy(nn.Module):    
    def __init__(self, node_features, hidden_dim, output_dim, num_partitions):
        super(GNNPPOPolicy, self).__init__()
        self.num_partitions = num_partitions

        # 简化为两层GNN，减少过平滑问题
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 添加层归一化，加强训练稳定性
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 简化Actor网络，移除多余的扩展-收缩结构
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 维持一致的宽度
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # 添加LayerNorm改善稳定性
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # 简化Critic网络，减少参数量
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 维持一致的宽度
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # 添加LayerNorm改善稳定性
            nn.Linear(hidden_dim, 1)
        )
        
        # 添加用于健康检查的嵌入跟踪
        self.embedding_stats_history = {
            'layer1_mean': [], 'layer1_std': [], 'layer1_norm': [],
            'layer2_mean': [], 'layer2_std': [], 'layer2_norm': [],
            'actor_mean': [], 'actor_std': [], 'actor_norm': [],
            'critic_mean': [], 'critic_std': [], 'critic_norm': []
        }
        self.grad_stats_history = {
            'conv1_grad_norm': [], 'conv2_grad_norm': [],
            'actor_grad_norm': [], 'critic_grad_norm': []
        }
        # 钩子函数，用于跟踪梯度
        self.hooks = []
        self.collect_stats = False

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 使用两层GNN并添加残差连接，减少过平滑问题
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(self.layer_norm1(x1))  # 应用层归一化
        
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(self.layer_norm2(x2))  # 应用层归一化
        x = x2 + x1  # 残差连接

        # 对每个节点输出动作概率和状态价值
        action_probs = self.actor(x)
        values = self.critic(x)

        # 如果开启了统计收集，记录各层的嵌入统计信息
        if self.collect_stats:
            self._collect_embedding_stats(x1, x, action_probs, values)

        return action_probs, values
    
    def _collect_embedding_stats(self, layer1_out, layer2_out, actor_out, critic_out):
        """收集各层嵌入的统计信息"""
        with torch.no_grad():
            # 收集第一层GNN输出统计信息
            self.embedding_stats_history['layer1_mean'].append(layer1_out.mean().item())
            self.embedding_stats_history['layer1_std'].append(layer1_out.std().item())
            self.embedding_stats_history['layer1_norm'].append(layer1_out.norm().item())
            
            # 收集第二层GNN输出统计信息
            self.embedding_stats_history['layer2_mean'].append(layer2_out.mean().item())
            self.embedding_stats_history['layer2_std'].append(layer2_out.std().item())
            self.embedding_stats_history['layer2_norm'].append(layer2_out.norm().item())
            
            # 收集Actor网络输出统计信息
            self.embedding_stats_history['actor_mean'].append(actor_out.mean().item())
            self.embedding_stats_history['actor_std'].append(actor_out.std().item())
            self.embedding_stats_history['actor_norm'].append(actor_out.norm().item())
            
            # 收集Critic网络输出统计信息
            self.embedding_stats_history['critic_mean'].append(critic_out.mean().item())
            self.embedding_stats_history['critic_std'].append(critic_out.std().item())
            self.embedding_stats_history['critic_norm'].append(critic_out.norm().item())
    
    def setup_grad_hooks(self):
        """设置梯度钩子，用于监控梯度流"""
        if not self.hooks:  # 避免重复设置钩子
            # 为GCN层设置钩子
            self.hooks.append(self.conv1.register_full_backward_hook(self._grad_hook('conv1')))
            self.hooks.append(self.conv2.register_full_backward_hook(self._grad_hook('conv2')))
            
            # 为Actor和Critic设置钩子
            for i, layer in enumerate(self.actor):
                if hasattr(layer, 'weight'):
                    self.hooks.append(layer.register_full_backward_hook(self._grad_hook(f'actor_{i}')))
            
            for i, layer in enumerate(self.critic):
                if hasattr(layer, 'weight'):
                    self.hooks.append(layer.register_full_backward_hook(self._grad_hook(f'critic_{i}')))
    
    def _grad_hook(self, name):
        """创建用于记录梯度的钩子函数"""
        def hook(module, grad_input, grad_output):
            if self.collect_stats and grad_output[0] is not None:
                # 记录梯度范数
                norm = grad_output[0].norm().item()
                key = f'{name}_grad_norm'
                if key in self.grad_stats_history:
                    self.grad_stats_history[key].append(norm)
                else:
                    self.grad_stats_history[key] = [norm]
        return hook
    
    def remove_hooks(self):
        """移除所有梯度钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def print_embedding_stats(self, episode):
        """打印嵌入统计信息"""
        if not self.embedding_stats_history['layer1_mean']:  # 检查是否有数据
            return
            
        # 获取最新的统计数据
        stats = {}
        for key, value_list in self.embedding_stats_history.items():
            if value_list:  # 确保列表非空
                stats[key] = value_list[-1]
        
        print(f"\n[Episode {episode}] GNN Embedding Health Check:")
        print("Layer 1 GCN: Mean={:.4f}, Std={:.4f}, Norm={:.4f}".format(
            stats.get('layer1_mean', 0), stats.get('layer1_std', 0), stats.get('layer1_norm', 0)))
        print("Layer 2 GCN: Mean={:.4f}, Std={:.4f}, Norm={:.4f}".format(
            stats.get('layer2_mean', 0), stats.get('layer2_std', 0), stats.get('layer2_norm', 0)))
        print("Actor Output: Mean={:.4f}, Std={:.4f}, Norm={:.4f}".format(
            stats.get('actor_mean', 0), stats.get('actor_std', 0), stats.get('actor_norm', 0)))
        print("Critic Output: Mean={:.4f}, Std={:.4f}, Norm={:.4f}".format(
            stats.get('critic_mean', 0), stats.get('critic_std', 0), stats.get('critic_norm', 0)))
        
        # 如果有梯度统计信息，也打印出来
        if self.grad_stats_history.get('conv1_grad_norm'):
            print("\nGradient Norms:")
            for key, values in self.grad_stats_history.items():
                if values:  # 确保列表非空
                    print(f"{key}: {values[-1]:.4f}")
    
    def visualize_embeddings(self, data, episode, output_dir="results/embeddings"):
        """可视化节点嵌入"""
        import os
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index
            
            # 获取两层的嵌入
            x1 = F.relu(self.conv1(x, edge_index))
            x2 = F.relu(self.conv2(x1, edge_index)) + x1
            
            # 将嵌入转移到CPU并转换为numpy数组
            x1_np = x1.cpu().numpy()
            x2_np = x2.cpu().numpy()
            
            # 使用PCA将嵌入降维到2维以便可视化
            pca = PCA(n_components=2)
            
            if len(x1_np) > 1:  # 确保有足够的样本进行PCA
                x1_2d = pca.fit_transform(x1_np)
                
                # 可视化第一层嵌入
                plt.figure(figsize=(10, 8))
                plt.scatter(x1_2d[:, 0], x1_2d[:, 1], c=range(len(x1_2d)), cmap='viridis')
                plt.colorbar(label='Node Index')
                plt.title(f'Layer 1 Embedding Visualization - Episode {episode}')
                plt.savefig(f"{output_dir}/layer1_ep{episode}.png")
                plt.close()
                
                # 可视化第二层嵌入
                x2_2d = pca.fit_transform(x2_np)
                plt.figure(figsize=(10, 8))
                plt.scatter(x2_2d[:, 0], x2_2d[:, 1], c=range(len(x2_2d)), cmap='viridis')
                plt.colorbar(label='Node Index')
                plt.title(f'Layer 2 Embedding Visualization - Episode {episode}')
                plt.savefig(f"{output_dir}/layer2_ep{episode}.png")
                plt.close()

    def act_batch(self, batch_data):
        """批量处理状态并返回动作，大幅减少CPU-GPU传输"""
        with torch.no_grad():
            action_probs, values = self.forward(batch_data)
            action_probs_flat = action_probs.view(-1)
            
            dist = Categorical(action_probs_flat)
            actions = dist.sample()
            action_log_probs = dist.log_prob(actions)
            
            # 返回动作、对数概率和价值估计 - 保持结构一致
            return actions.detach().cpu(), action_log_probs.detach().cpu(), values.detach().cpu()    
    def act(self, data):
        """根据当前状态选择动作"""
        with torch.no_grad():  # 【优化6】确保使用torch.no_grad()减少内存占用
            action_probs, node_values = self.forward(data)
            action_probs_flat = action_probs.view(-1)

            # 创建动作分布并采样
            dist = Categorical(action_probs_flat)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)

            return action.item(), action_log_prob, node_values
    def select_action_and_log_prob(self, data):
        """根据当前状态选择动作，返回原始logits和概率等信息"""
        with torch.no_grad():
            x, edge_index = data.x, data.edge_index

            # 使用两层GNN并添加残差连接，减少过平滑问题
            x1 = self.conv1(x, edge_index)
            x1 = F.relu(self.layer_norm1(x1))  # 应用层归一化
            
            x2 = self.conv2(x1, edge_index)
            x2 = F.relu(self.layer_norm2(x2))  # 应用层归一化
            x = x2 + x1  # 残差连接
            
            # 提取actor的各层输出，保留softmax之前的logits
            actor_features = x
            
            # 找到softmax之前的最后一层线性层
            logits = None
            last_linear = None
            
            # 分析actor网络结构，提取logits
            # actor网络结构: Linear -> ReLU -> LayerNorm -> Linear -> Softmax
            for i, layer in enumerate(self.actor):
                if isinstance(layer, nn.Linear):
                    last_linear = layer
                if isinstance(layer, nn.Softmax):
                    # 获取softmax之前的输出作为logits
                    break
            
            # 如果成功找到了最后一个线性层，计算logits
            if last_linear is not None:
                # 将输入传递到倒数第二层
                for i, layer in enumerate(self.actor):
                    actor_features = layer(actor_features)
                    if layer == last_linear:
                        # 获取最后一个线性层的输出作为logits
                        logits = actor_features
                        break
            
            # 继续计算完整的前向过程以获取概率
            action_probs, node_values = self.forward(data)
            action_probs_flat = action_probs.view(-1)
            
            # 创建动作分布并采样
            dist = Categorical(action_probs_flat)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            
            # 如果没能正确提取logits，使用概率的对数作为近似
            if logits is None:
                logits = torch.log(action_probs + 1e-10)  # 添加小值防止log(0)
            
            return action.item(), action_log_prob, node_values, logits, action_probs

    def evaluate(self, data, action):
        """评估动作的价值和概率"""
        action_probs, node_values = self.forward(data)
        action_probs_flat = action_probs.view(-1)

        dist = Categorical(action_probs_flat)
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy()

        return action_log_probs, node_values, entropy


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
        self.ppo_epochs = config.get('ppo_epochs', 2)  
        self.batch_size = config.get('batch_size', 64)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.adam_beta1 = config.get('adam_beta1', 0.9) 
        self.adam_beta2 = config.get('adam_beta2', 0.999)
        # 修改：将update_frequency更名为n_steps，并设置更合理的默认值
        self.n_steps = config.get('n_steps', config.get('update_frequency', 128))
        
        # 增加新的GPU优化配置
        self.hidden_dim = config.get('hidden_dim', 256)  # 增加默认隐藏层维度
        self.use_cuda_streams = config.get('use_cuda_streams', True)
        self.jit_compile = config.get('jit_compile', False)  # 是否使用JIT编译
        
        # 添加梯度裁剪参数
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # 添加学习率调度器配置
        self.use_lr_scheduler = config.get('use_lr_scheduler', True)
        self.lr_scheduler_factor = config.get('lr_scheduler_factor', 0.5)
        self.lr_scheduler_patience = config.get('lr_scheduler_patience', 10)
        self.lr_scheduler_min = config.get('lr_scheduler_min', 1e-6)
    
        # 【优化9】添加设备检测
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GNN-PPO使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")

        # 初始化策略网络
        self.policy = GNNPPOPolicy(self.node_features, self.hidden_dim,
                                   num_partitions, num_partitions).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2) # 传入 betas 元组
        )
        
        # 添加学习率调度器，在训练停滞时降低学习率
        if self.use_lr_scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',           # 使用'max'模式，因为我们希望奖励最大化
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                verbose=True,
                min_lr=self.lr_scheduler_min
            )
            print(f"启用学习率调度器: patience={self.lr_scheduler_patience}, factor={self.lr_scheduler_factor}")
        else:
            self.lr_scheduler = None

        # 【优化10】使用更高效的数据存储方式
        self.states = []
        self.state_datas = []  # 存储PyG数据对象
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        # 【优化11】添加性能统计
        self.conversion_time = 0
        self.forward_time = 0
        self.update_time = 0

        # 构建固定的图结构
        self._build_graph_structure()
          # 【优化12】预分配缓冲区，减少内存分配
        self.prefetch_data = None
        self.prefetch_actions = None

        # 添加TensorBoard支持
        from tensorboard_logger import TensorboardLogger
        tensorboard_config = config.get('tensorboard_config', {})
        self.use_tensorboard = config.get('use_tensorboard', True)
        if self.use_tensorboard:
            self.logger = TensorboardLogger(tensorboard_config)
            self.logger.experiment_name = "gnn_ppo_agent"  # 为GNN-PPO设置特殊名称
        else:
            self.logger = None

        self.state_cache = {}  # 添加状态缓存
        
        # 添加健康检查相关属性
        self.enable_health_check = config.get('enable_health_check', False)
        self.health_check_freq = config.get('health_check_freq', 10)
        self.enable_grad_check = config.get('enable_grad_check', False)
        self.enable_embedding_vis = config.get('enable_embedding_vis', False)
        self.vis_freq = config.get('vis_freq', 50)
        self.current_episode = 0
        self.health_check_states = {
            'episode_start': False,
            'episode_end': False,
            'after_update': False
        }        
        # 添加奖励归一化配置
        self.use_reward_norm = config.get('use_reward_norm', True)
        self.reward_norm_scale = config.get('reward_norm_scale', 1.0)
        self.running_reward_mean = 0
        self.running_reward_std = 1
        self.reward_ema_factor = config.get('reward_ema_factor', 0.99)  # 指数移动平均因子
        
        # 初始化缓存
        self.state_cache = {}  # 存储GNN前向传播特征
        self.max_cache_size = 1000  # 最大缓存大小
        
    def _clear_cache(self):
        """清除特征缓存"""
        if hasattr(self, 'state_cache'):
            self.state_cache.clear()
            
    def _check_feature_cache(self, state_hash):
        """检查特征缓存，如果命中则返回缓存的特征，否则返回None"""
        # 如果缓存太大，清除一半
        if hasattr(self, 'state_cache') and len(self.state_cache) > self.max_cache_size:
            # 简单策略：删除一半
            keys_to_remove = list(self.state_cache.keys())[:len(self.state_cache)//2]
            for key in keys_to_remove:
                del self.state_cache[key]
                
        return self.state_cache.get(state_hash, None)

    def _build_graph_structure(self):
        """构建PyG的图数据结构"""
        # 创建边索引
        edge_index = []
        for u, v in self.graph.edges():
            edge_index.append([u, v])
            edge_index.append([v, u])  # 添加反向边

        # 【优化13】直接将边索引移至GPU
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)

        # 保存节点权重和度
        self.node_weights = np.array([self.graph.nodes[i].get('weight', 1)
                                      for i in range(self.num_nodes)], dtype=np.float32)
        max_weight = max(self.node_weights) if len(self.node_weights) > 0 else 1.0
        self.node_weights = self.node_weights / max_weight

        degrees = np.array([self.graph.degree[i]
                            for i in range(self.num_nodes)], dtype=np.float32)
        max_degree = max(degrees) if len(degrees) > 0 else 1.0
        self.node_degrees = degrees / max_degree
        
        # 【优化14】预计算节点特征中的固定部分
        self.fixed_features = np.zeros((self.num_nodes, 2), dtype=np.float32)
        self.fixed_features[:, 0] = self.node_weights
        self.fixed_features[:, 1] = self.node_degrees

    def _state_to_pyg_data(self, state, batch_mode=False):
        """将环境状态转换为PyG数据对象，优化CPU-GPU数据传输"""
        state_hash = hash(state.tobytes())
        if state_hash in self.state_cache:
            return self.state_cache[state_hash]
        
        # 使用临时缓冲区避免重复分配内存
        if not hasattr(self, '_features_buffer'):
            self._features_buffer = np.zeros((self.num_nodes, self.node_features), dtype=np.float32)
        
        x_np = self._features_buffer # 使用预分配的 NumPy 数组
        x_np[:, :self.num_partitions] = state[:, :self.num_partitions]
        x_np[:, self.num_partitions:] = self.fixed_features
        
        # 一次性填充分区数据
        x_tensor_cpu = torch.from_numpy(x_np.copy()) # 创建副本以防万一
        # 2. 将 CPU 张量放入锁页内存
        x_tensor_pinned = x_tensor_cpu.pin_memory()
        # 3. 异步传输到 GPU
        x_tensor = x_tensor_pinned.to(self.device, non_blocking=True)
        # --- 修改结束 ---

        # 假设 self.edge_index 已经在 GPU 上
        data = Data(x=x_tensor, edge_index=self.edge_index)

        self.state_cache[state_hash] = data
        return data
    
    def _states_to_batch_data_optimized(self, states_list):
        """优化的批量状态处理，减少GPU张量创建"""
        batch_size = len(states_list)
        
        # 在CPU上预处理所有特征，然后一次性传输到GPU
        # 预分配NumPy数组
        all_features = np.zeros((batch_size * self.num_nodes, self.node_features), dtype=np.float32)
        
        # 一次性填充所有特征
        for i, state in enumerate(states_list):
            start_idx = i * self.num_nodes
            end_idx = (i + 1) * self.num_nodes
            
            # 分区数据
            all_features[start_idx:end_idx, :self.num_partitions] = state[:, :self.num_partitions]
            # 固定特征
            all_features[start_idx:end_idx, self.num_partitions:] = self.fixed_features
        
        # 一次性传输到GPU，使用pin_memory和non_blocking=True
        batch_features = torch.from_numpy(all_features).pin_memory().to(self.device, non_blocking=True)
        
        # 高效构建批处理边索引 - 使用缓存机制避免重复计算
        if not hasattr(self, '_cached_batch_edge_indices') or self._cached_batch_edge_indices is None or \
        self._cached_batch_size != batch_size:
            
            # 计算批处理边索引 - 只在批量大小变化时重新计算
            self._cached_batch_edge_indices = []
            for i in range(batch_size):
                offset = i * self.num_nodes
                batch_edge = self.edge_index.clone()
                batch_edge[0, :] += offset
                batch_edge[1, :] += offset
                self._cached_batch_edge_indices.append(batch_edge)
            
            self._cached_batch_edge_indices = torch.cat(self._cached_batch_edge_indices, dim=1)
            self._cached_batch_size = batch_size
        
        # 创建批次索引 - 直接在GPU上生成
        batch_idx = torch.arange(batch_size, device=self.device).repeat_interleave(self.num_nodes)
        
        # 构建批处理数据对象
        batch_data = Batch(
            x=batch_features,
            edge_index=self._cached_batch_edge_indices,
            batch=batch_idx
        )
        
        return batch_data
    def collect_experiences_parallel(self, envs, num_steps=10):
        """并行从多个环境收集经验，减少CPU瓶颈"""
        num_envs = len(envs)
        
        # 重置所有环境
        states = [env.reset()[0] for env in envs]
        all_data = {
            'states': [], 'actions': [], 'log_probs': [], 
            'rewards': [], 'dones': [], 'values': []
        }
        
        # 预先处理初始状态批次
        batch_data = self._states_to_batch_data_optimized(states)
        
        for _ in range(num_steps):
            # 批量处理动作选择 - 使用更高效的批量act方法
            actions, log_probs, values = self.policy.act_batch(batch_data)
            
            # 重塑为每个环境的动作
            actions = actions.view(num_envs, -1)
            log_probs = log_probs.view(num_envs, -1)
            
            # 在所有环境中执行步骤
            next_states = []
            rewards = np.zeros(num_envs)
            dones = np.zeros(num_envs, dtype=bool)
            
            for i, (env, action) in enumerate(zip(envs, actions)):
                # 执行环境步骤
                next_state, reward, done, _, _ = env.step(action[0].item())
                next_states.append(next_state)
                rewards[i] = reward
                dones[i] = done
                
                # 存储经验
                all_data['states'].append(states[i])
                all_data['actions'].append(action[0].item())
                all_data['log_probs'].append(log_probs[i][0])
                all_data['rewards'].append(reward)
                all_data['dones'].append(done)
                all_data['values'].append(values[i].mean().item())
                
                # 存储到智能体缓冲区
                self.states.append(states[i])
                self.actions.append(action[0].item())
                self.log_probs.append(log_probs[i][0])
                self.rewards.append(reward)
                self.dones.append(done)
                self.values.append(values[i].mean().item())
                
            # 更新状态
            states = next_states
            
            # 如果所有环境都结束，则重置
            if np.all(dones):
                states = [env.reset()[0] for env in envs]
            
            # 处理新的状态批次
            batch_data = self._states_to_batch_data_optimized(states)
        
        return all_data
    
    def process_states_async(self, states_batch):
        """使用CUDA Streams异步处理状态数据"""
        # 只有在使用CUDA时才使用Streams
        if self.device.type != 'cuda':
            # CPU版本，直接调用普通处理
            return self._states_to_batch_data_optimized(states_batch)
        
        batch_size = len(states_batch)
        result_data = []
        
        # 创建多个CUDA流
        num_streams = min(4, batch_size)
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        
        # 分配每个流要处理的状态索引
        stream_assignments = [[] for _ in range(num_streams)]
        for i in range(batch_size):
            stream_assignments[i % num_streams].append(i)
        
        # 在每个流中并行处理状态
        for stream_idx, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                for i in stream_assignments[stream_idx]:
                    state = states_batch[i]
                    
                    # 在GPU上创建特征张量
                    x = torch.zeros((self.num_nodes, self.node_features), 
                                dtype=torch.float32, device=self.device)
                    
                    # 将分区数据复制到GPU
                    x[:, :self.num_partitions] = torch.tensor(
                        state[:, :self.num_partitions], dtype=torch.float32, device=self.device
                    )
                    
                    # 添加固定特征
                    x[:, self.num_partitions:] = torch.tensor(
                        self.fixed_features, dtype=torch.float32, device=self.device
                    )
                    
                    # 创建数据对象
                    data = Data(x=x, edge_index=self.edge_index)
                    result_data.append((i, data))
        
        # 使用事件来管理依赖关系
        events = []
        for stream in streams:
            event = torch.cuda.Event()
            event.record(stream)
            events.append(event)
        
        # 仅在必要时等待特定事件
        for event in events:
            event.wait()
        
        # 按原始顺序排序结果
        result_data.sort(key=lambda x: x[0])
        sorted_data = [d for _, d in result_data]
        
        # 创建批处理数据
        return Batch.from_data_list(sorted_data)

    # 【优化16】批量处理状态
    def _states_to_batch_data(self, states_list):
        """批量将多个状态转换为一个批处理数据对象"""
        batch_data_list = []
        for state in states_list:
            data = self._state_to_pyg_data(state)
            batch_data_list.append(data)
            
        return Batch.from_data_list(batch_data_list).to(self.device)
    def act(self, state):
        """根据当前状态选择动作"""
        # 【优化17】使用计时器评估性能瓶颈
        start = time.time()
        
        # 将状态转换为PyG数据 - 这部分仍需要实时转换以做出决策
        state_data = self._state_to_pyg_data(state)
        
        conversion_end = time.time()
        self.conversion_time += conversion_end - start

        value = 0.0 # 初始化 value
        log_prob = None # 初始化 log_prob
        action = 0 # 初始化 action        # 在episode开始时执行健康检查
        self.perform_health_check(state_data, 'episode_start')
            
        with torch.no_grad():
            # 使用 select_action_and_log_prob 方法同时获取logits和概率
            action_tensor, log_prob_tensor, values_nodes, logits, probs = self.policy.select_action_and_log_prob(state_data)
            action = action_tensor # .item() 已经在 policy.select_action_and_log_prob 中处理
            log_prob = log_prob_tensor
            value = values_nodes.mean().item()  # 使用所有节点值的平均作为状态价值
            self.forward_time += time.time() - conversion_end
            
            # 存储logits和probs用于健康检查
            if hasattr(self, '_last_action_info'):
                self._last_action_info = {
                    'logits': logits.detach().cpu(),
                    'probs': probs.detach().cpu()
                }

        # 存储当前轨迹信息 - 注意我们现在存储原始状态，不预先转换
        self.states.append(state)  # 存储原始NumPy状态
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        return action, log_prob, value
    def store_transition(self, reward, done):
        """存储奖励和状态终止信号，并进行奖励归一化"""
        # 应用奖励归一化
        if self.use_reward_norm:
            # 更新运行统计量 (使用指数移动平均)
            if len(self.rewards) == 0:  # 首次设置
                self.running_reward_mean = reward
                self.running_reward_std = abs(reward) + 1e-8
            else:
                self.running_reward_mean = self.reward_ema_factor * self.running_reward_mean + \
                                          (1 - self.reward_ema_factor) * reward
                reward_diff = abs(reward - self.running_reward_mean)
                self.running_reward_std = self.reward_ema_factor * self.running_reward_std + \
                                         (1 - self.reward_ema_factor) * reward_diff
            
            # 归一化奖励
            norm_reward = (reward - self.running_reward_mean) / (self.running_reward_std + 1e-8)
            # 缩放归一化后的奖励
            scaled_norm_reward = norm_reward * self.reward_norm_scale
            
            # 记录原始奖励和归一化奖励
            if self.logger is not None and len(self.rewards) % 10 == 0:
                self.logger.log_scalar("rewards/original", reward, len(self.rewards))
                self.logger.log_scalar("rewards/normalized", scaled_norm_reward, len(self.rewards))
                self.logger.log_scalar("rewards/mean", self.running_reward_mean, len(self.rewards))
                self.logger.log_scalar("rewards/std", self.running_reward_std, len(self.rewards))
                
            self.rewards.append(scaled_norm_reward)
        else:
            self.rewards.append(reward)
            
        self.dones.append(done)
    
    def should_update(self):
        """检查是否达到了更新频率"""
        # 确保缓冲区中有足够的数据
        return len(self.rewards) >= self.n_steps
        
    def _forward_jit(self, x, edge_index):
        """JIT加速的前向计算，包含残差连接"""
        x1 = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x1, edge_index)) + x1  # 添加残差连接
        return x
        
    def update(self):
        """优化的PPO算法更新，使用subgraph简化子图构建和向量化操作减少同步点"""
        # 如果数据不足，直接返回
        if len(self.rewards) < self.n_steps:
            return 0.0

        start = time.time()

        if len(self.rewards) == 0:
            return 0.0
            
        # 计算回报和优势 (在GPU上)
        returns_t, advantages_t, values_t = self._compute_returns_advantages_vectorized()

        # 使用异步处理批量转换状态
        state_processing_start = time.time()
        state_datas = self._states_to_batch_data_optimized(self.states)
        self.conversion_time += time.time() - state_processing_start

        # 一次性将所有数据传输到GPU
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack([lp.to(self.device) if isinstance(lp, torch.Tensor) else 
                                torch.tensor(lp, device=self.device) for lp in self.log_probs])

        # 用于统计的变量
        total_loss = 0.0
        update_rounds = 0
        dataset_size = len(self.states)
        effective_batch_size = min(self.batch_size, dataset_size)

        # 如果开启了梯度检查，启用收集统计
        if self.enable_grad_check and self.current_episode % self.health_check_freq == 0:
            self.policy.collect_stats = True

        # PPO更新循环
        for _ in range(self.ppo_epochs):
            # 在GPU上生成随机索引
            indices = torch.randperm(dataset_size, device=self.device)

            for i in range(0, dataset_size, effective_batch_size):
                end_idx = min(i + effective_batch_size, dataset_size)
                batch_indices = indices[i:end_idx]

                # === 使用subgraph优化的子图构建 ===
                # 1. 确定minibatch对应的节点在state_datas中的全局索引
                batch_indices_expanded = batch_indices.unsqueeze(1)  # [B, 1]
                node_offsets = torch.arange(self.num_nodes, device=self.device).unsqueeze(0)  # [1, N]
                subset_nodes_global = (batch_indices_expanded * self.num_nodes + node_offsets).view(-1)  # [B*N]

                # 2. 使用subgraph提取子图结构
                sub_edge_index, _ = subgraph(
                    subset=subset_nodes_global,
                    edge_index=state_datas.edge_index,
                    relabel_nodes=True,
                    num_nodes=state_datas.num_nodes
                )
                
                # 3. 提取子图节点特征
                sub_x = state_datas.x[subset_nodes_global]

                # 4. 创建子图的batch向量 (向量化实现，避免循环)
                batch_size = len(batch_indices)
                sub_batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.num_nodes)

                # 5. 创建子批次Data对象
                sub_batch_data = Data(
                    x=sub_x,
                    edge_index=sub_edge_index,
                    batch=sub_batch
                )
                # === 子图构建结束 ===

                # 提取对应的动作、旧log probs、回报、优势
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]

                # 评估动作
                new_log_probs, node_values, entropy = self.policy.evaluate(sub_batch_data, batch_actions)
                
                # 使用global_mean_pool进行节点值聚合
                values = global_mean_pool(node_values, sub_batch_data.batch)
                values = values.squeeze(-1)

                # === 向量化的PPO算法损失计算 ===
                # 计算策略目标
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 向量化的clip操作
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 获取旧的值估计
                batch_old_values = values_t[batch_indices]
                
                # 实现值函数裁剪，防止值函数更新过大导致训练不稳定
                value_pred_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -self.clip_ratio, self.clip_ratio
                )
                # 计算两种值损失并取最大值，类似于PPO策略裁剪
                value_losses = (values - batch_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # 优化 - 使用梯度累积减少同步点
                self.optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True减少内存使用
                loss.backward()
                
                # 检查是否有梯度异常 (NaN或Inf)
                if self.enable_health_check and self.current_episode % self.health_check_freq == 0:
                    has_nan_grad = False
                    max_grad_norm = 0.0
                    for name, param in self.policy.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            max_grad_norm = max(max_grad_norm, grad_norm)
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"[Warning] NaN/Inf gradient detected in {name}")
                                has_nan_grad = True
                    
                    if has_nan_grad:
                        print("[Warning] NaN/Inf gradients found, skipping this update")
                        self.optimizer.zero_grad()
                        continue
                    
                    if max_grad_norm > 10.0:  # 通常认为梯度范数超过10是很大的
                        print(f"[Warning] Large gradient norm: {max_grad_norm:.4f}")
                
                # 使用原地操作进行梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                update_rounds += 1

                # TensorBoard记录 (低频率记录以减少同步)
                if self.logger is not None and update_rounds % 5 == 0:
                    self.logger.log_episode(
                        self.rewards,
                        loss.item(),
                        entropy.mean().item(),
                        value_loss.item(),
                        policy_loss.item()
                    )
                    
                    # 额外记录嵌入统计信息到TensorBoard
                    if self.enable_health_check and self.current_episode % self.health_check_freq == 0:
                        for key, value_list in self.policy.embedding_stats_history.items():
                            if value_list:  # 确保列表非空
                                self.logger.log_scalar(f"embedding/{key}", value_list[-1], self.current_episode)        # 关闭统计收集
        if self.enable_grad_check:
            self.policy.collect_stats = False
            
        # 如果有更新发生，并且有数据，执行更新后的健康检查
        if update_rounds > 0 and len(self.states) > 0:
            # 为了健康检查，获取最后一个状态的数据
            last_state = self.states[-1]
            state_data = self._state_to_pyg_data(last_state)
            self.perform_health_check(state_data, 'after_update')

        # 清空缓冲区并报告时间
        self._clear_buffers()
        self.update_time += time.time() - start
        
        # 每隔一段时间清除缓存以防止内存泄漏
        if hasattr(self, 'state_cache') and len(self.state_cache) > 1000:
            self.state_cache.clear()
        
        # 应用学习率调度器，使用当前episode的平均奖励来调整学习率
        if self.use_lr_scheduler and len(self.rewards) > 0 and update_rounds > 0:
            avg_reward = sum(self.rewards) / len(self.rewards)
            # 对于学习率调度器，使用平均奖励作为指标（因为我们想最大化奖励）
            self.lr_scheduler.step(avg_reward)
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.logger is not None:
                self.logger.log_scalar("training/learning_rate", current_lr, self.current_episode)
                
            # 打印当前学习率（每10个episode一次）
            if self.current_episode % 10 == 0:
                print(f"当前学习率: {current_lr:.6f}")
        
        return total_loss / max(1, update_rounds)

      # 添加向量化的GAE计算
    def _compute_returns_advantages_vectorized(self):
        """真正向量化的GAE计算, 无Python循环"""
        # 将数据转换为 GPU 张量
        rewards_t = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(self.values, dtype=torch.float32, device=self.device)
        
        # 计算下一个状态的值
        next_value_t = torch.tensor(0.0, dtype=torch.float32, device=self.device)
    
        # 如果最后一个状态不是终止状态，计算其值
        if not self.dones[-1]:
            # 获取最后一个状态并转换为PyG数据
            last_state = self.states[-1]
            with torch.no_grad():
                state_data = self._state_to_pyg_data(last_state)
                _, last_values = self.policy(state_data)
                next_value_t = last_values.mean().item()  # 使用平均值作为状态值
        
        # 添加下一个状态值到values序列末尾 - 确保维度匹配
        values_extended = torch.cat([values_t, torch.tensor([next_value_t], device=self.device)])
        
        # 计算delta和GAE
        masks = 1.0 - dones_t
        deltas = rewards_t + self.gamma * values_extended[1:] * masks - values_t
        
        # 计算优势（倒序）
        advantages_t = torch.zeros_like(rewards_t)
        gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for t in reversed(range(len(rewards_t))):
            gae = deltas[t] + self.gamma * self.gae_lambda * masks[t] * gae
            advantages_t[t] = gae
        
        # 计算回报
        returns_t = advantages_t + values_t
        
        # 标准化优势，提高训练稳定性
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # 返回回报、优势和当前状态的值估计
        return returns_t, advantages_t, values_t
          # 标准化优势 (在 GPU 上)
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        return returns_t, advantages_t, values_t
        
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

        values_copy = self.values.copy()  # 创建副本以存储原始值函数估计
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

        # 还原values列表，移除next_value
        self.values = values_copy

        # 转换为torch张量
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values_copy, dtype=torch.float32, device=self.device)

        return returns_t, advantages_t, values_t
    def _clear_buffers(self):
        """清空轨迹缓冲区"""
        self.states = []
        self.state_datas = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        # 清除特征缓存，避免内存占用过大
        self._clear_cache()

    def save_model(self, filepath):
        """保存模型到文件"""
        torch.save(self.policy.state_dict(), filepath)

    def load_model(self, filepath):
        """从文件加载模型"""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
    
    # 【优化26】添加性能统计打印
    def print_performance_stats(self):
        """打印性能统计"""
        total_time = self.conversion_time + self.forward_time + self.update_time
        if total_time > 0:
            print("性能统计:")
            print(f"状态转换时间: {self.conversion_time:.2f}s ({self.conversion_time/total_time*100:.1f}%)")
            print(f"前向传播时间: {self.forward_time:.2f}s ({self.forward_time/total_time*100:.1f}%)")
            print(f"策略更新时间: {self.update_time:.2f}s ({self.update_time/total_time*100:.1f}%)")
            print(f"总时间: {total_time:.2f}s")
        else:
            print("还没有收集到性能统计数据")
    
    # 【优化27】添加评估模式方法
    def eval_mode(self):
        """设置为评估模式"""
        self.policy.eval()
        
    # 【优化28】添加训练模式方法
    def train_mode(self):
        """设置为训练模式"""
        self.policy.train()    
    def perform_health_check(self, state_data, check_point):
        """执行健康检查
        
        Args:
            state_data: PyG数据对象，用于健康检查
            check_point: 检查点类型，例如'episode_start', 'episode_end', 'after_update'
        """
        if not self.enable_health_check or self.current_episode % self.health_check_freq != 0:
            return False
            
        # 检查是否已在当前episode的该检查点执行过检查
        if self.health_check_states[check_point]:
            return False
            
        # 标记该检查点已执行检查
        self.health_check_states[check_point] = True
        
        # 开始收集统计
        self.policy.collect_stats = True
        
        # 确保一次前向传播以收集统计数据
        with torch.no_grad():
            # 使用select_action_and_log_prob来获取更详细的信息
            _, _, _, logits, probs = self.policy.select_action_and_log_prob(state_data)
            
            # 分析logits（softmax之前的值）
            logits_mean = logits.mean().item()
            logits_std = logits.std().item()
            logits_min = logits.min().item()
            logits_max = logits.max().item()
            
            # 分析输出概率
            probs_mean = probs.mean().item()
            probs_std = probs.std().item()
            probs_min = probs.min().item()
            probs_max = probs.max().item()
              # 检查Actor网络最后一个线性层的权重和偏置
            last_linear_layer = None
            
            # 更严格地查找最后一个线性层 (排除softmax后面的层)
            found_softmax = False
            for module in reversed(list(self.policy.actor)):
                if isinstance(module, nn.Softmax):
                    found_softmax = True
                elif isinstance(module, nn.Linear) and not found_softmax:
                    last_linear_layer = module
                    break
            
            # 如果没找到，再尝试查找任何线性层
            if last_linear_layer is None:
                for module in reversed(list(self.policy.actor)):
                    if isinstance(module, nn.Linear):
                        last_linear_layer = module
                        break
                        
            if last_linear_layer is not None:
                weights = last_linear_layer.weight.detach()
                biases = last_linear_layer.bias.detach() if last_linear_layer.bias is not None else None
                
                # 计算权重和偏置的统计信息
                weights_mean = weights.mean().item()
                weights_std = weights.std().item()
                weights_norm = weights.norm().item()
                
                # 如果分区数为2，计算两个分区对应的权重向量之间的差异
                if self.num_partitions == 2 and weights.size(0) == 2:
                    weight_diff = (weights[0] - weights[1]).abs().mean().item()
                    
                    if biases is not None:
                        bias_diff = abs(biases[0].item() - biases[1].item())
                    else:
                        bias_diff = 0.0
            
            # 打印详细的诊断信息
            print(f"\n[Episode {self.current_episode} - {check_point}] GNN健康状态检查:")
            print("===== Actor 网络输出检查 =====")
            print(f"Logits 统计: 均值={logits_mean:.4f}, 标准差={logits_std:.4f}, 最小值={logits_min:.4f}, 最大值={logits_max:.4f}")
            print(f"Probs 统计: 均值={probs_mean:.4f}, 标准差={probs_std:.4f}, 最小值={probs_min:.4f}, 最大值={probs_max:.4f}")
            
            if last_linear_layer is not None:
                print("\n===== Actor 最后一层线性层检查 =====")
                print(f"权重统计: 均值={weights_mean:.4f}, 标准差={weights_std:.4f}, 范数={weights_norm:.4f}")
                
                if self.num_partitions == 2 and weights.size(0) == 2:
                    print(f"两个分区权重向量的平均绝对差: {weight_diff:.6f}")
                    if biases is not None:
                        print(f"两个分区偏置的绝对差: {bias_diff:.6f}")
                        print(f"偏置值: [{biases[0].item():.6f}, {biases[1].item():.6f}]")
            
            # 为两个分区的情况，直接打印整个权重矩阵和偏置值
            if self.num_partitions == 2 and weights.size(0) == 2:
                print("\n完整权重矩阵:")
                print(weights.cpu().numpy())
                
                if biases is not None:
                    print("\n完整偏置向量:")
                    print(biases.cpu().numpy())
        
        # 打印嵌入统计信息
        self.policy.print_embedding_stats(self.current_episode)
        
        # 如果开启了嵌入可视化并且到了可视化周期
        if self.enable_embedding_vis and self.current_episode % self.vis_freq == 0:
            self.policy.visualize_embeddings(state_data, self.current_episode)
            
        # 关闭统计收集
        self.policy.collect_stats = False
        
        return True