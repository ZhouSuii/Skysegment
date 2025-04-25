# agent ppo
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


# PPO策略网络模型
class PPOPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PPOPolicy, self).__init__()
        # 优化点1: 合并actor网络层减少计算深度
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  # 移除一个隐藏层
            nn.Softmax(dim=-1)
        )
        
        # 优化点2: 合并critic网络层减少计算深度
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 移除一个隐藏层
        )

    def forward(self, x):
        # 优化点3: 使用jit.script加速前向传播
        # 此处添加@torch.jit.script注解可进一步加速，但可能需要调整代码
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value

    # 优化点4: 优化act方法，支持批处理
    def act_batch(self, states):
        """批量处理状态并返回动作"""
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)
        actions = dist.sample()
        action_log_probs = dist.log_prob(actions)
        return actions.detach(), action_log_probs.detach(), values.detach()

    def act(self, state):
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob

    def evaluate(self, state, action):
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return action_log_probs, value.squeeze(-1), entropy


# PPO智能体
class PPOAgent:
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size

        # 加载配置或使用默认值
        if config is None:
            config = {}

        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.learning_rate = config.get('learning_rate', 0.0003)
        
        # 优化点5: 减少PPO更新轮数，大幅度提高速度
        self.ppo_epochs = config.get('ppo_epochs', 4)  # 从10降到4
        self.batch_size = config.get('batch_size', 64)  # 增大批量
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        
        # 优化点6: 添加更新频率控制，减少更新次数
        self.update_frequency = config.get('update_frequency', 4)  # 每4个episode更新一次
        
        # 优化点7: 使用更紧凑的缓冲区，并预分配
        self.memory_capacity = config.get('memory_capacity', 10000) # Add memory capacity config
        self.states_buffer = np.zeros((self.memory_capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(self.memory_capacity, dtype=np.int64)
        self.log_probs = np.zeros(self.memory_capacity, dtype=np.float32)
        self.rewards = np.zeros(self.memory_capacity, dtype=np.float32)
        self.dones = np.zeros(self.memory_capacity, dtype=np.float32)
        self.values = np.zeros(self.memory_capacity, dtype=np.float32)
        
        # 轨迹计数和缓冲区指针
        self.buffer_ptr = 0
        self.traj_start_ptr = 0 # Track start of the current trajectory
        
        # 检查GPU可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO使用设备: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")

        # 初始化策略网络并移动到GPU
        self.policy = PPOPolicy(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # 优化点8: 使用pin_memory提前将数据pin到内存中，加速数据迁移至GPU
        if self.device.type == 'cuda':
            self.pin_memory = True
        else:
            self.pin_memory = False

        # tensorboard
        from tensorboard_logger import TensorboardLogger
        tensorboard_config = config.get('tensorboard_config', {})
        self.use_tensorboard = config.get('use_tensorboard', True)
        if self.use_tensorboard:
            self.logger = TensorboardLogger(tensorboard_config)
        else:
            self.logger = None

    def act(self, state):
        """根据当前状态选择动作"""
        # 将状态展平并转移到GPU
        state_tensor = torch.FloatTensor(np.array(state).flatten()).unsqueeze(0).to(self.device) # Add unsqueeze(0) for batch dim
        
        self.policy.eval() # Set to eval mode
        with torch.no_grad():
            action, log_prob = self.policy.act(state_tensor)
            _, value = self.policy(state_tensor)
        self.policy.train() # Set back to train mode

        # 存储当前轨迹信息到 NumPy 缓冲区
        if self.buffer_ptr < self.memory_capacity:
            self.states_buffer[self.buffer_ptr] = np.array(state).flatten()
            self.actions[self.buffer_ptr] = action
            # Ensure log_prob is a scalar float
            self.log_probs[self.buffer_ptr] = log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob 
            self.values[self.buffer_ptr] = value.item()
        else:
            # Handle buffer overflow (e.g., overwrite oldest or raise error)
            print("Warning: PPO buffer overflow!")
            # Simple overwrite for now:
            self.buffer_ptr = 0 
            self.states_buffer[self.buffer_ptr] = np.array(state).flatten()
            self.actions[self.buffer_ptr] = action
            self.log_probs[self.buffer_ptr] = log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob
            self.values[self.buffer_ptr] = value.item()

        return action

    def store_transition(self, reward, done):
        """存储奖励和状态终止信号，并推进缓冲区指针"""
        if self.buffer_ptr < self.memory_capacity:
            self.rewards[self.buffer_ptr] = reward
            self.dones[self.buffer_ptr] = float(done)
            self.buffer_ptr += 1 # Advance pointer after storing transition

    def update(self):
        """使用PPO算法更新策略"""
        # Calculate number of steps collected since last update
        steps_collected = self.buffer_ptr - self.traj_start_ptr
        
        # 优化点10: 根据更新频率决定是否执行更新
        # Check if enough steps collected OR if buffer is full
        if steps_collected < self.update_frequency and self.buffer_ptr < self.memory_capacity: 
            return 0.0
            
        # Ensure there are steps to process
        if steps_collected <= 0:
             self.traj_start_ptr = self.buffer_ptr # Reset trajectory start if no steps
             return 0.0

        # --- Data Preparation --- 
        # Slice the relevant part of the buffer
        indices = np.arange(self.traj_start_ptr, self.buffer_ptr)
        states_np = self.states_buffer[indices]
        actions_np = self.actions[indices]
        old_log_probs_np = self.log_probs[indices]
        rewards_np = self.rewards[indices]
        dones_np = self.dones[indices]
        values_np = self.values[indices]

        # 计算优势估计和回报 (使用向量化版本)
        returns, advantages = self._compute_returns_advantages_vectorized(rewards_np, dones_np, values_np)

        # --- Data Transfer to GPU --- 
        # 使用 pin_memory 加速数据迁移至 GPU
        if self.pin_memory and self.device.type == 'cuda':
            states = torch.tensor(states_np, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
            actions = torch.tensor(actions_np, dtype=torch.long).pin_memory().to(self.device, non_blocking=True)
            old_log_probs = torch.tensor(old_log_probs_np, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
            returns = torch.tensor(returns, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
            advantages = torch.tensor(advantages, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
        else:
            states = torch.tensor(states_np, dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions_np, dtype=torch.long).to(self.device)
            old_log_probs = torch.tensor(old_log_probs_np, dtype=torch.float32).to(self.device)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        # --- PPO Update Loop --- 
        total_loss = 0.0
        update_rounds = 0
        dataset_size = len(indices)
        current_batch_size = min(self.batch_size, dataset_size)

        # 在GPU上生成随机索引
        perm_indices = torch.randperm(dataset_size, device=self.device)
        
        for _ in range(self.ppo_epochs):
            perm_indices = perm_indices[torch.randperm(len(perm_indices), device=self.device)] # Shuffle indices each epoch
            
            for start_idx in range(0, dataset_size, current_batch_size):
                end_idx = min(start_idx + current_batch_size, dataset_size)
                batch_indices = perm_indices[start_idx:end_idx]
                
                # Directly index GPU tensors
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate current policy
                new_log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)

                # Calculate ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Calculate PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Calculate value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Calculate entropy bonus
                entropy_mean = entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean
                total_loss += loss.item()

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                update_rounds += 1

        # --- Cleanup --- 
        # Reset the trajectory start pointer for the next collection phase
        self.traj_start_ptr = self.buffer_ptr
        # Handle buffer wrap-around
        if self.buffer_ptr == self.memory_capacity:
            self.traj_start_ptr = 0
            self.buffer_ptr = 0
            print("PPO buffer wrapped around.")

        # Return average loss for this update cycle
        return total_loss / max(1, update_rounds)
        
    # 优化点14: 修改为接收 NumPy 数组并返回 NumPy 数组
    def _compute_returns_advantages_vectorized(self, rewards, dones, values):
        """使用向量化方式计算GAE和回报，接收NumPy数组"""
        # rewards, dones, values are slices from the buffer for the trajectory
        
        # 计算下一个状态的值 (approximated by the value of the last state in trajectory)
        # If the trajectory didn't end (last done is False), bootstrap from the policy
        if dones[-1]:
            next_value = 0.0
        else:
            with torch.no_grad():
                # Get the very last state stored (which is state for the next action)
                last_state_idx = self.buffer_ptr - 1
                if last_state_idx < 0: last_state_idx = self.memory_capacity - 1 # wrap around
                state_tensor = torch.FloatTensor(self.states_buffer[last_state_idx]).unsqueeze(0).to(self.device)
                _, value_tensor = self.policy(state_tensor)
                next_value = value_tensor.item()
        
        # 添加下一个状态值到values序列末尾 (for calculation)
        values_with_next = np.append(values, next_value)
        
        # 预分配优势数组
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # GAE计算
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_with_next[t + 1] * (1 - dones[t]) - values_with_next[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        # 计算回报 = 优势 + 值
        returns = advantages + values # values is already the correct slice V(s_t)
        
        # 标准化优势 (only if more than one step)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

    # Modify _clear_buffers to just reset pointers
    def _clear_buffers(self):
        """清空轨迹缓冲区 (by resetting pointers)"""
        self.buffer_ptr = 0
        self.traj_start_ptr = 0
        # No need to zero out NumPy arrays, they will be overwritten

    def save_model(self, filepath):
        """保存模型到文件"""
        torch.save(self.policy.state_dict(), filepath)
        if self.logger is not None:
            self.logger.close()

    def load_model(self, filepath):
        """从文件加载模型"""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        
    # 优化点15: 添加设置为评估模式的方法
    def eval_mode(self):
        """设置为评估模式, 关闭dropout等训练特性"""
        self.policy.eval()
        
    # 优化点16: 添加设置为训练模式的方法
    def train_mode(self):
        """设置为训练模式, 启用dropout等训练特性"""
        self.policy.train()