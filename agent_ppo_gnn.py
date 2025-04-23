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

        # 增加 GNN 层数和宽度，提高 GPU 利用率
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # 新增第三层
        
        # 扩大 Actor 网络宽度
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

        # 扩大 Critic 网络宽度
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 增加更多计算层
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # 对每个节点输出动作概率和状态价值
        action_probs = self.actor(x)
        values = self.critic(x)

        return action_probs, values
    
    # 【优化5】添加批量动作选择功能
    def act_batch(self, batch_data):
        """批量处理状态并返回动作，大幅减少CPU-GPU传输"""
        with torch.no_grad():
            action_probs, values = self.forward(batch_data)
            action_probs_flat = action_probs.view(-1)
            
            dist = Categorical(action_probs_flat)
            actions = dist.sample()
            action_log_probs = dist.log_prob(actions)
            
            return actions.detach().cpu(), action_log_probs.detach().cpu(), values.detach().cpu()

    def act(self, data):
        """根据当前状态选择动作"""
        with torch.no_grad():  # 【优化6】确保使用torch.no_grad()减少内存占用
            action_probs, _ = self.forward(data)
            action_probs_flat = action_probs.view(-1)

            # 创建动作分布并采样
            dist = Categorical(action_probs_flat)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)

            return action.item(), action_log_prob

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
        
        # 【优化7】减少PPO更新轮数，提高速度
        self.ppo_epochs = config.get('ppo_epochs', 4)  # 从10降到4
        self.batch_size = config.get('batch_size', 64)  # 增大批处理大小
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # 【优化8】添加更新频率参数，减少更新次数
        self.update_frequency = config.get('update_frequency', 4) 
        self.update_counter = 0
        
        # 增加新的GPU优化配置
        self.hidden_dim = config.get('hidden_dim', 256)  # 增加默认隐藏层维度
        self.use_cuda_streams = config.get('use_cuda_streams', True)
        self.jit_compile = config.get('jit_compile', False)  # 是否使用JIT编译
    
        # 【优化9】添加设备检测
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GNN-PPO使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")

        # 初始化策略网络
        self.policy = GNNPPOPolicy(self.node_features, self.hidden_dim,
                                   num_partitions, num_partitions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

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
        # 使用临时缓冲区避免重复分配内存
        if not hasattr(self, '_features_buffer'):
            self._features_buffer = np.zeros((self.num_nodes, self.node_features), dtype=np.float32)
        
        # 直接使用预先分配的缓冲区
        x = self._features_buffer
        
        # 一次性填充分区数据
        x[:, :self.num_partitions] = state[:, :self.num_partitions]
        # 直接使用预计算的固定特征
        x[:, self.num_partitions:] = self.fixed_features
        
        # 一次性转换为Tensor并移至GPU
        x_tensor = torch.from_numpy(x.copy()).to(self.device)
        
        # 创建 PyG 数据对象，使用预先计算的边索引
        data = Data(x=x_tensor, edge_index=self.edge_index)

        return data
    
    def _states_to_batch_data_optimized(self, states_list):
        """优化的批量状态处理，直接在GPU上构建批处理数据"""
        batch_size = len(states_list)
        
        # 预分配整个批次的特征张量，直接在GPU上创建
        batch_features = torch.zeros(
            (batch_size * self.num_nodes, self.node_features),
            dtype=torch.float32, 
            device=self.device
        )
        
        # 一次性处理所有状态
        for i, state in enumerate(states_list):
            start_idx = i * self.num_nodes
            end_idx = (i + 1) * self.num_nodes
            
            # 分区数据直接复制到GPU
            batch_features[start_idx:end_idx, :self.num_partitions] = torch.tensor(
                state[:, :self.num_partitions], dtype=torch.float32, device=self.device
            )
            
            # 固定特征也直接在GPU上填充
            batch_features[start_idx:end_idx, self.num_partitions:] = torch.tensor(
                self.fixed_features, dtype=torch.float32, device=self.device
            )
        
        # 构建批处理图结构
        # 创建重复的边索引
        batch_edge_index = []
        for i in range(batch_size):
            offset = i * self.num_nodes
            # 复制边索引并添加偏移
            batch_edge = self.edge_index.clone()
            batch_edge[0, :] += offset
            batch_edge[1, :] += offset
            batch_edge_index.append(batch_edge)
        
        # 合并所有边索引
        batch_edge_index = torch.cat(batch_edge_index, dim=1)
        
        # 创建批次索引
        batch_idx = torch.arange(batch_size, device=self.device).repeat_interleave(self.num_nodes)
        
        # 构建批处理数据对象
        batch_data = Batch(
            x=batch_features,
            edge_index=batch_edge_index,
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
            # 批量处理动作选择
            with torch.no_grad():
                action_probs, values = self.policy.forward(batch_data)
                action_probs_flat = action_probs.view(-1)
                
                # 为每个节点采样动作
                dist = Categorical(action_probs_flat)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                
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
        
        # 等待所有流完成
        torch.cuda.synchronize()
        
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

        with torch.no_grad():
            action, log_prob = self.policy.act(state_data)
            _, values = self.policy(state_data)
            value = values.mean().item()  # 使用所有节点值的平均作为状态价值
                
        self.forward_time += time.time() - conversion_end

        # 存储当前轨迹信息 - 注意我们现在存储原始状态，不预先转换
        self.states.append(state)  # 存储原始NumPy状态
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        # 【优化18】增加更新计数
        self.update_counter += 1

        return action

    def store_transition(self, reward, done):
        """存储奖励和状态终止信号"""
        self.rewards.append(reward)
        self.dones.append(done)

    @torch.jit.script_if_tracing
    def _forward_jit(self, x, edge_index):
        """JIT加速的前向计算"""
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return x

    def update(self):
        """优化的PPO算法更新，提高GPU利用率"""
        # 检查是否需要更新
        if self.update_counter < self.update_frequency:
            return 0.0

        self.update_counter = 0
        start = time.time()
        
        # 确保有足够的数据
        if len(self.rewards) == 0:
            return 0.0

        # 计算回报和优势
        returns, advantages = self._compute_returns_advantages_vectorized()

        # 使用异步处理批量转换状态
        state_processing_start = time.time()
        state_datas = self.process_states_async(self.states)
        self.conversion_time += time.time() - state_processing_start

        # 一次性将所有数据传输到GPU
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack([lp.to(self.device) for lp in self.log_probs])
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # 使用更大的批次进行更新
        total_loss = 0.0
        update_rounds = 0
        dataset_size = len(self.states)
        
        # 增加批处理大小以提高 GPU 利用率
        effective_batch_size = min(self.batch_size * 2, dataset_size)

        # 使用 torch.randperm 在 GPU 上生成随机索引
        if self.device.type == 'cuda':
            indices = torch.randperm(dataset_size, device=self.device)
        else:
            indices = torch.randperm(dataset_size)

        # 多轮更新
        for _ in range(self.ppo_epochs):
            # 每轮重新打乱数据
            if self.device.type == 'cuda':
                indices = indices[torch.randperm(len(indices), device=self.device)]
            else:
                indices = indices[torch.randperm(len(indices))]
            
            # 批次处理
            for idx in range(0, dataset_size, effective_batch_size):
                end_idx = min(idx + effective_batch_size, dataset_size)
                batch_indices = indices[idx:end_idx]
                
                # 从大批量中提取子批量
                batch_states_data = []
                cpu_indices = batch_indices.cpu().numpy()
                for i in cpu_indices:
                    batch_states_data.append(state_datas[i])
                
                # 创建批处理数据
                batch_state_data = Batch.from_data_list(batch_states_data).to(self.device)
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # 评估动作
                new_log_probs, node_values, entropy = self.policy.evaluate(batch_state_data, batch_actions)
                
                # 聚合节点值
                values = global_mean_pool(node_values, batch_state_data.batch)
                values = values.squeeze(-1)

                # 计算策略目标
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 计算价值损失
                value_loss = F.mse_loss(values, batch_returns)

                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                update_rounds += 1

        # 清空缓冲区
        self._clear_buffers()
        
        self.update_time += time.time() - start
        
        # 返回平均损失
        return total_loss / max(1, update_rounds)
    
    # 【优化25】添加向量化的GAE计算
    def _compute_returns_advantages_vectorized(self):
        """使用向量化计算GAE提高性能"""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)
        
        # 计算下一个状态的值
        if self.dones[-1]:
            next_value = 0
        else:
            with torch.no_grad():
                next_state_data = self._state_to_pyg_data(self.states[-1])
                _, values_tensor = self.policy(next_state_data)
                next_value = values_tensor.mean().item()
        
        # 添加下一个状态值到values序列末尾
        values = np.append(values, next_value)
        
        # 预分配优势数组
        advantages = np.zeros_like(rewards)
        
        # 使用循环计算GAE (后续可以进一步向量化)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        # 计算回报
        returns = advantages + values[:-1]
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

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
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
    
    # 【优化26】添加性能统计打印
    def print_performance_stats(self):
        """打印性能统计信息"""
        total_time = self.conversion_time + self.forward_time + self.update_time
        if total_time > 0:
            print(f"性能统计:")
            print(f"  状态转换时间: {self.conversion_time:.2f}s ({self.conversion_time/total_time*100:.1f}%)")
            print(f"  前向传播时间: {self.forward_time:.2f}s ({self.forward_time/total_time*100:.1f}%)")
            print(f"  策略更新时间: {self.update_time:.2f}s ({self.update_time/total_time*100:.1f}%)")
            print(f"  总时间: {total_time:.2f}s")
    
    # 【优化27】添加评估模式方法
    def eval_mode(self):
        """设置为评估模式"""
        self.policy.eval()
        
    # 【优化28】添加训练模式方法
    def train_mode(self):
        """设置为训练模式"""
        self.policy.train()