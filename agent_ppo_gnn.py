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

        # 简化网络结构，减少过拟合和梯度消失
        self.conv1 = GCNConv(node_features, hidden_dim // 2)  # 减小隐藏层维度
        self.conv2 = GCNConv(hidden_dim // 2, hidden_dim // 2)
        
        # 移除LayerNorm，简化网络结构
        self.dropout = nn.Dropout(0.1)
        
        # 简化Actor网络，直接输出logits，移除Softmax
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, output_dim)
            # 移除Softmax，在外部计算
        )
        
        # 简化Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # 使用Xavier初始化改善权重初始化
        self.apply(self._init_weights)
        
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
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, GCNConv):
            # GCNConv内部有线性层，需要初始化
            if hasattr(module, 'lin') and module.lin is not None:
                nn.init.xavier_uniform_(module.lin.weight, gain=1.0)
                if hasattr(module.lin, 'bias') and module.lin.bias is not None:
                    nn.init.zeros_(module.lin.bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GNN前向传播
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.dropout(x1)
        x2 = F.relu(self.conv2(x1, edge_index)) 
        x2 = self.dropout(x2)
        
        # Actor和Critic输出
        logits = self.actor(x2)
        values = self.critic(x2)
        action_probs = F.softmax(logits, dim=-1)
        
        return logits, values, action_probs

    def evaluate(self, data, action):
        logits, values, action_probs = self.forward(data)
        action_probs_flat = action_probs.view(-1)
        dist = Categorical(action_probs_flat)
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return action_log_probs, values, entropy
    
    def _collect_embedding_stats(self, layer1_out, layer2_out, actor_out, critic_out):
        """修复统计信息收集，确保正确存储到embedding_stats_history中"""
        # 计算并直接存储统计信息到embedding_stats_history字典中
        self.embedding_stats_history['layer1_mean'].append(layer1_out.mean().item())
        self.embedding_stats_history['layer1_std'].append(layer1_out.std().item())
        self.embedding_stats_history['layer1_norm'].append(layer1_out.norm().item())
        
        self.embedding_stats_history['layer2_mean'].append(layer2_out.mean().item())
        self.embedding_stats_history['layer2_std'].append(layer2_out.std().item())
        self.embedding_stats_history['layer2_norm'].append(layer2_out.norm().item())
        
        self.embedding_stats_history['actor_mean'].append(actor_out.mean().item())
        self.embedding_stats_history['actor_std'].append(actor_out.std().item())
        self.embedding_stats_history['actor_norm'].append(actor_out.norm().item())
        
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

            # 使用简化的GNN结构（与forward方法保持一致）
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
            x = self.dropout(x)
            
            # 获取actor的logits（softmax之前的输出）
            logits = self.actor(x)
            
            # 计算动作概率
            action_probs = F.softmax(logits, dim=-1)
            action_probs_flat = action_probs.view(-1)
            
            # 获取critic的输出
            values = self.critic(x)
            
            # 创建动作分布并采样
            dist = Categorical(action_probs_flat)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            
            return action.item(), action_log_prob, values, logits, action_probs

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
        self.clip_ratio = config.get('clip_ratio', 0.1)
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.ppo_epochs = config.get('ppo_epochs', 2)  
        self.batch_size = config.get('batch_size', 64)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.hidden_dim = config.get('hidden_dim', 32)
        self.adam_beta1 = config.get('adam_beta1', 0.9) 
        self.adam_beta2 = config.get('adam_beta2', 0.999)
        # 修改：将update_frequency更名为n_steps，并设置更合理的默认值
        self.n_steps = config.get('n_steps', config.get('update_frequency', 128))
        
        # 增加新的GPU优化配置
        self.use_cuda_streams = config.get('use_cuda_streams', True)
        self.jit_compile = config.get('jit_compile', False)  # 是否使用JIT编译
        
        # 添加梯度裁剪参数
        self.max_grad_norm = config.get('max_grad_norm', 0.3)
        
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
        # 使用更保守的优化器设置
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            eps=1e-5,  # 增加epsilon提高数值稳定性
            weight_decay=1e-4  # 添加权重衰减
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
        """修复状态转换，确保输入特征非零并添加调试信息"""
        state_hash = hash(state.tobytes()) if not batch_mode else None
        if state_hash and state_hash in self.state_cache:
            return self.state_cache[state_hash]
        
        # 确保输入特征有足够的变化
        if not hasattr(self, '_features_buffer'):
            self._features_buffer = np.zeros((self.num_nodes, self.node_features), dtype=np.float32)
        
        x_np = self._features_buffer
        
        # 填充one-hot分区编码
        x_np[:, :self.num_partitions] = state[:, :self.num_partitions]
        
        # 添加节点权重和度特征（确保非零）
        x_np[:, self.num_partitions] = self.node_weights  # 原始权重
        x_np[:, self.num_partitions + 1] = self.node_degrees  # 归一化度
          # 转换为tensor并移动到GPU
        x_tensor = torch.from_numpy(x_np.copy()).to(self.device)
        
        # 创建数据对象
        data = Data(x=x_tensor, edge_index=self.edge_index)
        
        if state_hash:
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
        start = time.time()
        
        # 将状态转换为PyG数据
        state_data = self._state_to_pyg_data(state)
        
        conversion_end = time.time()
        self.conversion_time += conversion_end - start

        # 在episode开始时执行健康检查
        self.perform_health_check(state_data, 'episode_start')
            
        with torch.no_grad():
            action_probs, node_values = self.policy(state_data)
            action_probs_flat = action_probs.view(-1)

            # 创建动作分布并采样
            dist = Categorical(action_probs_flat)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            
            value = node_values.mean().item()  # 使用所有节点值的平均作为状态价值
            self.forward_time += time.time() - conversion_end

        # 存储当前轨迹信息
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(action_log_prob)
        self.values.append(value)
        
        return action.item(), action_log_prob, value

    def store_transition(self, reward, done):
        """修复奖励归一化，避免数值不稳定"""
        if self.use_reward_norm:
            # 使用更稳定的归一化方法
            if len(self.rewards) < 10:  # 前几步不进行归一化
                self.rewards.append(reward)
            else:
                # 使用滑动窗口计算统计量
                recent_rewards = self.rewards[-100:] + [reward]  # 最近100个奖励
                mean_reward = np.mean(recent_rewards)
                std_reward = np.std(recent_rewards) + 1e-8
                
                # 限制归一化的范围，避免极端值
                normalized_reward = np.clip((reward - mean_reward) / std_reward, -5, 5)
                self.rewards.append(normalized_reward)
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
        """修复价值函数聚合和梯度流问题"""
        if len(self.rewards) < self.n_steps:
            return 0.0

        start = time.time()
        
        # 计算回报和优势
        returns_t, advantages_t, values_t = self._compute_returns_advantages_vectorized()

        # 批量转换状态
        state_datas = self._states_to_batch_data_optimized(self.states)

        # 一次性将所有数据传输到GPU
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack([lp.to(self.device) if isinstance(lp, torch.Tensor) else 
                                torch.tensor(lp, device=self.device) for lp in self.log_probs])

        total_loss = 0.0
        update_rounds = 0
        dataset_size = len(self.states)
        effective_batch_size = min(self.batch_size, dataset_size)        # 保留训练数据大小但不打印详细信息
        dataset_size = len(self.states)

        # PPO更新循环
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(dataset_size, device=self.device)

            for i in range(0, dataset_size, effective_batch_size):
                end_idx = min(i + effective_batch_size, dataset_size)
                batch_indices = indices[i:end_idx]

                # 构建子图
                batch_indices_expanded = batch_indices.unsqueeze(1)
                node_offsets = torch.arange(self.num_nodes, device=self.device).unsqueeze(0)
                subset_nodes_global = (batch_indices_expanded * self.num_nodes + node_offsets).view(-1)

                sub_edge_index, _ = subgraph(
                    subset=subset_nodes_global,
                    edge_index=state_datas.edge_index,
                    relabel_nodes=True,
                    num_nodes=state_datas.num_nodes
                )
                
                sub_x = state_datas.x[subset_nodes_global]
                batch_size = len(batch_indices)
                sub_batch = torch.arange(batch_size, device=self.device).repeat_interleave(self.num_nodes)

                sub_batch_data = Data(
                    x=sub_x,
                    edge_index=sub_edge_index,
                    batch=sub_batch
                )

                # 提取批次数据
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]

                # 评估动作
                new_log_probs, node_values, entropy = self.policy.evaluate(sub_batch_data, batch_actions)
                
                # 修复价值聚合 - 确保每个样本对应一个价值
                if node_values.dim() == 2 and node_values.size(0) == batch_size * self.num_nodes:
                    # 重塑并计算每个图的平均值
                    node_values_reshaped = node_values.view(batch_size, self.num_nodes, -1)
                    values = node_values_reshaped.mean(dim=1).squeeze(-1)
                else:
                    # 使用global_mean_pool作为备选
                    values = global_mean_pool(node_values, sub_batch_data.batch)
                    values = values.squeeze(-1)
                
                # 确保维度匹配
                assert values.size(0) == batch_size, f"Value size mismatch: {values.size(0)} vs {batch_size}"                # PPO损失计算
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 简化价值损失计算
                value_loss = F.mse_loss(values, batch_returns)

                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # 检查损失的数值稳定性
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[错误] 检测到NaN/Inf损失: {loss.item()}")
                    continue
                
                # 优化
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                # 记录梯度统计
                total_grad_norm = 0
                for name, param in self.policy.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2                
                        total_grad_norm = total_grad_norm ** (1. / 2)
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                update_rounds += 1

        # 清空缓冲区
        self._clear_buffers()
        self.update_time += time.time() - start
        
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
        """确保正确收集和显示嵌入统计信息"""
        if not self.enable_health_check or self.current_episode % self.health_check_freq != 0:
            return False
            
        # 检查是否已在当前episode的该检查点执行过检查
        if self.health_check_states.get(check_point, False):
            return False
            
        # 标记该检查点已执行检查
        self.health_check_states[check_point] = True
        
        # 激活统计收集
        self.policy.collect_stats = True
        
        # 强制进行一次前向传播以收集统计数据
        with torch.no_grad():
            action_probs, values = self.policy.forward(state_data)
        
        # 打印嵌入统计信息
        self.policy.print_embedding_stats(self.current_episode)
        
        # 关闭统计收集
        self.policy.collect_stats = False
        
        return True