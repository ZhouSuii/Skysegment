# run experiments
import os
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import json
import torch
from tqdm import tqdm
import random

# 配置中文字体支持
try:
    # 尝试使用文泉驿微米黑字体
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 验证字体加载
    plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体
except Exception as e:
    print(f"字体配置错误: {e}，将使用默认字体")

from new_environment import GraphPartitionEnvironment
from agent_dqn_basic import DQNAgent
from metrics import evaluate_partition
from baselines import random_partition, weighted_greedy_partition, metis_partition, spectral_partition
from agent_gnn import GNNDQNAgent
from agent_ppo import PPOAgent
from agent_ppo_gnn_simple import SimplePPOAgentGNN
from metrics import calculate_weight_variance, calculate_partition_weights

def set_seed(seed=42):
    """
    设置所有随机种子，确保每个算法都从相同的随机状态开始
    这是解决算法间公平对比的关键函数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量确保Python hash的确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"🔧 重置随机种子为: {seed}")

def create_test_graph(num_nodes=10, seed=42):
    """创建测试图，带有节点权重"""
    np.random.seed(seed)
    G = nx.random_geometric_graph(num_nodes, 0.5, seed=seed)

    # 添加节点权重
    for i in range(num_nodes):
        G.nodes[i]['weight'] = np.random.randint(1, 10)

    return G


def load_graph_from_file(filepath):
    """从文件加载图"""
    try:
        # 根据文件扩展名选择加载方法
        extension = os.path.splitext(filepath)[1]
        if extension == '.graphml':
            G = nx.read_graphml(filepath)
        elif extension == '.gexf':
            G = nx.read_gexf(filepath)
        elif extension == '.edgelist':
            G = nx.read_edgelist(filepath)
        else:
            # 默认尝试pickle加载
            G = nx.read_gpickle(filepath)

        # 如果没有节点权重，添加默认权重1
        for node in G.nodes():
            if 'weight' not in G.nodes[node]:
                G.nodes[node]['weight'] = 1

        return G
    except Exception as e:
        print(f"加载图文件时出错: {e}")
        return None


def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        return {}


def train_dqn_agent(graph, num_partitions, config, results_dir="results"):
    # 每个算法都从相同的随机状态开始
    set_seed(42)
    """训练DQN智能体"""
    # 获取配置参数
    episodes = config.get("episodes", 1000)
    max_steps = config.get("max_steps", 100)
    batch_size = config.get("dqn_config", {}).get("batch_size", config.get("batch_size", 32))
    dqn_config = config.get("dqn_config", {}) # 获取DQN配置

    # --- 修改：使用 new_environment 并传递参数 ---
    # 使用默认的 potential_weights 或从配置加载
    default_potential_weights = {'variance': 1.0, 'edge_cut': 1.0, 'modularity': 1.0}
    potential_weights = config.get("potential_weights", default_potential_weights)
    env = GraphPartitionEnvironment(
        graph,
        num_partitions,
        max_steps,
        gamma=dqn_config.get('gamma', 0.95), # 从DQN配置获取gamma
        potential_weights=potential_weights
    )

    # 初始化DQN代理
    num_nodes = len(graph.nodes())
    # --- 修改：更新状态大小计算 (+1 for node weights) ---
    state_size = num_nodes * (num_partitions + 2)
    # --- 修改结束 ---
    action_size = num_nodes * num_partitions
    dqn_config['batch_size'] = batch_size # 传递正确的batch_size给agent
    agent = DQNAgent(state_size, action_size, dqn_config)

    best_reward = float('-inf')
    best_partition = None
    rewards_history = []
    loss_history = []  # 新增：记录损失历史
    variance_history = []  # 新增：记录方差历史
    
    # 新增: 按 episode 记录的列表
    episode_rewards = []
    episode_variances = []

    # 训练循环
    progress_bar = tqdm(range(episodes), desc="训练DQN")
    for e in progress_bar:
        state, _ = env.reset()
        total_reward = 0
        done = False

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                break
        
        # 使用 agent.memory_counter 获取实际存储的样本数
        memory_size = min(agent.memory_counter, agent.memory_capacity)
        
        # 计算当前 episode 的方差
        current_variance = calculate_weight_variance(graph, env.partition_assignment, num_partitions)

        # 进行经验回放学习
        if memory_size >= batch_size:
            loss = agent.replay(batch_size)  # 获取返回的损失值
            
            # 仅在有效更新时记录所有指标
            loss_history.append(loss)
            rewards_history.append(total_reward)
            variance_history.append(current_variance)

        # 按 episode 记录核心指标
        episode_rewards.append(total_reward)
        episode_variances.append(current_variance)

        # 更新最佳划分
        if total_reward > best_reward:
            best_reward = total_reward
            best_partition = env.partition_assignment.copy()

        # 更新进度条
        progress_bar.set_postfix({
            'reward': total_reward,
            'best': best_reward,
            'epsilon': agent.epsilon,
            'loss': loss_history[-1] if loss_history else 0,
            'variance': variance_history[-1] if variance_history else 0
        })

        # 每50个episode保存一次模型
        if (e + 1) % 50 == 0:
            os.makedirs(f"{results_dir}/models", exist_ok=True)
            agent.save_model(f"{results_dir}/models/dqn_model_{len(graph.nodes())}nodes_{num_partitions}parts_temp.pt")

    # 保存最终模型
    os.makedirs(f"{results_dir}/models", exist_ok=True)
    agent.save_model(f"{results_dir}/models/dqn_model_{len(graph.nodes())}nodes_{num_partitions}parts.pt")

    return best_partition, rewards_history, loss_history, variance_history, episode_rewards, episode_variances


def train_gnn_agent(graph, num_partitions, config, results_dir="results"):
    # 每个算法都从相同的随机状态开始
    set_seed(42)
    """训练GNN智能体"""
    # 获取配置参数
    episodes = config.get("episodes", 500)
    max_steps = config.get("max_steps", 100)
    batch_size = config.get("batch_size", 32)
    gnn_config = config.get("gnn_config", {}) # 获取GNN配置

    # --- 修改：使用 new_environment 并传递参数 ---
    default_potential_weights = {'variance': 1.0, 'edge_cut': 1.0, 'modularity': 1.0}
    potential_weights = config.get("potential_weights", default_potential_weights)
    env = GraphPartitionEnvironment(
        graph,
        num_partitions,
        max_steps,
        gamma=gnn_config.get('gamma', 0.95), # 从GNN配置获取gamma
        potential_weights=potential_weights
    )
    # --- 修改结束 ---

    # 初始化GNN-DQN代理
    agent = GNNDQNAgent(graph, num_partitions, gnn_config)

    best_reward = float('-inf')
    best_partition = None
    rewards_history = []
    loss_history = []  # 新增：记录损失历史
    variance_history = []  # 新增：记录方差历史
    
    # 新增: 按 episode 记录的列表
    episode_rewards = []
    episode_variances = []

    # 训练循环
    progress_bar = tqdm(range(episodes), desc="训练GNN-DQN")
    for e in progress_bar:
        state, _ = env.reset()
        total_reward = 0
        done = False

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 计算当前 episode 的方差
        current_variance = calculate_weight_variance(graph, env.partition_assignment, num_partitions)

        # 进行经验回放学习
        memory_size = min(agent.memory_counter, agent.memory_capacity)
        if memory_size >= batch_size:
            loss = agent.replay() # 获取返回的损失值
            
            # 仅在有效更新时记录所有指标
            loss_history.append(loss)
            rewards_history.append(total_reward)
            variance_history.append(current_variance)

        # 按 episode 记录核心指标
        episode_rewards.append(total_reward)
        episode_variances.append(current_variance)

        # 更新最佳划分
        if total_reward > best_reward:
            best_reward = total_reward
            best_partition = env.partition_assignment.copy()

        # 更新进度条
        progress_bar.set_postfix({
            'reward': total_reward,
            'best': best_reward,
            'epsilon': agent.epsilon,
            'loss': loss_history[-1] if loss_history else 0,
            'variance': variance_history[-1] if variance_history else 0
        })

    # 保存模型
    os.makedirs(f"{results_dir}/models", exist_ok=True)
    agent.save_model(f"{results_dir}/models/gnn_dqn_model_{len(graph.nodes())}nodes_{num_partitions}parts.pt")
    
    return best_partition, rewards_history, loss_history, variance_history, episode_rewards, episode_variances


def train_ppo_agent(graph, num_partitions, config, results_dir="results"):
    # 每个算法都从相同的随机状态开始
    set_seed(42)
    """训练PPO智能体"""
    # 获取配置参数
    episodes = config.get("episodes", 500)
    max_steps = config.get("max_steps", 100)
    ppo_config = config.get("ppo_config", {}) # 获取PPO配置

    # --- 修改：使用 new_environment 并传递参数 ---
    default_potential_weights = {'variance': 1.0, 'edge_cut': 1.0, 'modularity': 1.0}
    potential_weights = config.get("potential_weights", default_potential_weights)
    env = GraphPartitionEnvironment(
        graph,
        num_partitions,
        max_steps,
        gamma=ppo_config.get('gamma', 0.99), # 从PPO配置获取gamma
        potential_weights=potential_weights
    )
    # --- 修改结束 ---

    # 初始化PPO代理
    num_nodes = len(graph.nodes())
    # --- 修改：更新状态大小计算 (+1 for node weights) ---
    state_size = num_nodes * (num_partitions + 2)
    # --- 修改结束 ---
    action_size = num_nodes * num_partitions
    agent = PPOAgent(state_size, action_size, ppo_config)

    best_reward = float('-inf')
    best_partition = None
    rewards_history = []
    loss_history = []
    variance_history = [] # 新增

    # 新增: 按 episode 记录的列表
    episode_rewards = []
    episode_variances = []
    
    # 训练循环
    progress_bar = tqdm(range(episodes), desc="训练PPO")
    for e in progress_bar:
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_rewards = []

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.store_transition(reward, done)

            state = next_state
            total_reward += reward
            step_rewards.append(reward)

            # 记录单步信息
            if agent.logger is not None and e % agent.logger.log_freq == 0:
                agent.logger.log_scalar("step/reward", reward, agent.logger.step_count)
                agent.logger.step_count += 1

            if done:
                break
        
        # 计算当前 episode 的方差
        current_variance = calculate_weight_variance(graph, env.partition_assignment, num_partitions)

        # 更新策略
        loss = agent.update()
        if loss is not None:
            rewards_history.append(total_reward)
            loss_history.append(loss)
            variance_history.append(current_variance)
        
        # 按 episode 记录核心指标
        episode_rewards.append(total_reward)
        episode_variances.append(current_variance)

        # 记录额外指标
        if agent.logger is not None:
            metrics = {
                "performance/weight_variance": current_variance,
                "performance/total_steps": step + 1
            }
            agent.logger.log_metrics(metrics, e)

        # 更新进度条
        progress_bar.set_postfix({
            'reward': total_reward,
            'best': best_reward,
            'loss': loss_history[-1] if loss_history else 0,
            'variance': variance_history[-1] if variance_history else 0
        })

        # 保存最佳结果
        if total_reward > best_reward:
            best_reward = total_reward
            best_partition = env.partition_assignment.copy()

    # 保存模型
    os.makedirs(f"{results_dir}/models", exist_ok=True)
    agent.save_model(f"{results_dir}/models/ppo_model_{len(graph.nodes())}nodes_{num_partitions}parts.pt")

    return best_partition, rewards_history, loss_history, variance_history, episode_rewards, episode_variances


# 添加训练GNN-PPO智能体的函数
def train_gnn_ppo_agent(graph, num_partitions, config, results_dir="results"):
    # 每个算法都从相同的随机状态开始
    set_seed(42)
    """训练GNN-PPO智能体"""
    # 获取配置参数
    episodes = config.get("episodes", 500)
    max_steps = config.get("max_steps", 100)
    gnn_ppo_config = config.get("gnn_ppo_config", {}) # 获取GNN-PPO配置
    
    # === 删除：所有健康检查相关配置 ===
    # 设置分区数量参数，这是新接口需要的
    gnn_ppo_config['num_partitions'] = num_partitions

    # --- 修改：使用 new_environment 并传递参数 ---
    default_potential_weights = {'variance': 1.0, 'edge_cut': 1.0, 'modularity': 1.0}
    potential_weights = config.get("potential_weights", default_potential_weights)
    env = GraphPartitionEnvironment(
        graph,
        num_partitions,
        max_steps,
        gamma=gnn_ppo_config.get('gamma', 0.99), # 从GNN-PPO配置获取gamma
        potential_weights=potential_weights
    )
    # --- 修改结束 ---

    # === 修改：初始化新的GNN-PPO代理 ===
    # 计算节点特征维度：分区数 + 度 + 权重
    node_feature_dim = num_partitions + 2
    action_size = len(graph.nodes()) * num_partitions
    agent = SimplePPOAgentGNN(node_feature_dim, action_size, gnn_ppo_config)
    
    # 创建结果目录
    os.makedirs(f"{results_dir}/models", exist_ok=True)

    best_reward = float('-inf')
    best_partition = None
    rewards_history = []
    loss_history = []
    variance_history = []
    
    # 新增: 按 episode 记录的列表
    episode_rewards = []
    episode_variances = []

    # 训练循环    
    progress_bar = tqdm(range(episodes), desc="训练GNN-PPO")
    for e in progress_bar:
        # === 修改：使用图结构数据格式重置环境 ===
        graph_state, _ = env.reset(state_format='graph')
        total_reward = 0
        
        for step in range(max_steps):
            # === 修改：使用图数据进行动作选择 ===
            action = agent.act(graph_state)
            next_state, reward, done, _, _ = env.step(action)
            
            # === 修改：获取下一个状态的图数据格式 ===
            next_graph_state = env.get_state('graph')

            agent.store_transition(reward, done)
            
            graph_state = next_graph_state
            total_reward += reward
            if done:
                break

        # 计算当前 episode 的方差
        current_variance = np.var(calculate_partition_weights(graph, env.partition_assignment, num_partitions))
        
        # 更新策略
        loss, updated = agent.update()
        
        # 仅在有效更新时记录
        if updated:
            loss_history.append(loss)
            rewards_history.append(total_reward)
            variance_history.append(current_variance)
        
        # 按 episode 记录核心指标
        episode_rewards.append(total_reward)
        episode_variances.append(current_variance)

        # === 删除：所有健康检查相关代码 ===
        
        # 每50个episodes保存一次模型快照
        if e > 0 and e % 50 == 0:
            snapshot_path = f"{results_dir}/models/gnn_ppo_snapshot_ep{e}.pt"
            agent.save_model(snapshot_path)
            print(f"\n保存模型快照到 {snapshot_path}")

        # 更新进度条
        progress_bar.set_postfix({
            'reward': total_reward,
            'best': best_reward,
            'loss': loss_history[-1] if loss_history else 0,
            'variance': variance_history[-1] if variance_history else 0
        })

        # 保存最佳结果
        if total_reward > best_reward:
            best_reward = total_reward
            best_partition = env.partition_assignment.copy()

    # === 删除：性能统计打印 ===

    # 保存模型
    agent.save_model(f"{results_dir}/models/gnn_ppo_model_{len(graph.nodes())}nodes_{num_partitions}parts.pt")

    return best_partition, rewards_history, loss_history, variance_history, episode_rewards, episode_variances


def run_experiment(graph_name, graph, num_partitions, config, results_dir="results"):
    """运行一次完整的实验，包括所有算法"""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)

    results = {}
    methods = config.get("methods", ["random", "greedy", "spectral", "metis", "dqn", "gnn", "ppo", "gnn_ppo"])

    # 用于存储训练历史
    training_data = {}

    # 运行每种算法
    for method in methods:
        print(f"\n执行 {method} 算法:")
        start_time = time.time()

        if method == "random":
            partition = random_partition(graph, num_partitions)
        elif method == "greedy":
            partition = weighted_greedy_partition(graph, num_partitions)
        elif method == "spectral":
            partition = spectral_partition(graph, num_partitions)
        elif method == "metis":
            partition = metis_partition(graph, num_partitions)
        elif method == "dqn":
            partition, rewards, losses, variances, episode_rewards, episode_variances = train_dqn_agent(graph, num_partitions, config, results_dir)
            # 记录训练历史
            training_data["dqn"] = {
                "rewards": rewards,
                "loss": losses,
                "variance": variances,
                "episode_rewards": episode_rewards,
                "episode_variances": episode_variances
            }
        elif method == "gnn":
            partition, rewards, losses, variances, episode_rewards, episode_variances = train_gnn_agent(graph, num_partitions, config, results_dir)
            # 记录训练历史
            training_data["gnn"] = {
                "rewards": rewards,
                "loss": losses,
                "variance": variances,
                "episode_rewards": episode_rewards,
                "episode_variances": episode_variances
            }
        elif method == "ppo":
            partition, rewards, losses, variances, episode_rewards, episode_variances = train_ppo_agent(graph, num_partitions, config, results_dir)
            # 记录训练历史
            training_data["ppo"] = {
                "rewards": rewards,
                "loss": losses,
                "variance": variances,
                "episode_rewards": episode_rewards,
                "episode_variances": episode_variances
            }
        elif method == "gnn_ppo":
            partition, rewards, losses, variances, episode_rewards, episode_variances = train_gnn_ppo_agent(graph, num_partitions, config, results_dir)
            # 记录训练历史
            training_data["gnn_ppo"] = {
                "rewards": rewards,
                "loss": losses,
                "variance": variances,
                "episode_rewards": episode_rewards,
                "episode_variances": episode_variances
            }
        else:
            print(f"未知方法: {method}")
            continue

        end_time = time.time()
        runtime = end_time - start_time

        # 评估分区质量
        eval_results = evaluate_partition(graph, partition, num_partitions, print_results=True)
        eval_results["runtime"] = runtime
        results[method] = eval_results

    # 保存结果为DataFrame
    results_df = pd.DataFrame()

    for method, eval_results in results.items():
        method_results = {
            "method": method,
            "weight_variance": eval_results["weight_variance"],
            "weight_imbalance": eval_results["weight_imbalance"],
            "edge_cut": eval_results["edge_cut"],
            "normalized_cut": eval_results["normalized_cut"],
            "modularity": eval_results["modularity"],
            "runtime": eval_results["runtime"]
        }
        results_df = pd.concat([results_df, pd.DataFrame([method_results])], ignore_index=True)

    # 保存结果到CSV
    results_df.to_csv(f"{results_dir}/{graph_name}_results.csv", index=False)
    print(f"结果已保存到 {results_dir}/{graph_name}_results.csv")

    # 绘制训练历史曲线图
    if training_data:
        plot_episode_curves(graph_name, training_data, results_dir)
        plot_avg_episode_curves_with_std(graph_name, training_data, results_dir)

    # 绘制比较图
    plot_comparison(graph_name, results, results_dir)

    return results_df


# 添加新的绘图函数
def plot_episode_curves(graph_name, training_data, results_dir):
    """绘制按 episode 记录的训练历史曲线图"""
    rl_methods = [m for m in training_data.keys()]
    if not rl_methods:
        return

    # 绘制奖励曲线 (按 Episode)
    plt.figure(figsize=(12, 8))
    for method in rl_methods:
        rewards = training_data[method].get("episode_rewards")
        if not rewards: continue

        window_size = min(10, len(rewards) // 10) if len(rewards) > 10 else 1
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        x_axis = np.arange(len(smoothed_rewards)) + (window_size - 1)
        
        plt.plot(x_axis, smoothed_rewards, label=method.upper())
        
    plt.title(f"Episodic Reward Curve - {graph_name}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()    
    plt.grid(True)
    plt.savefig(f"{results_dir}/plots/{graph_name}_episodic_rewards.png")
    plt.savefig(f"{results_dir}/plots/{graph_name}_episodic_rewards.svg", format='svg')
    plt.close()

    # 绘制方差曲线 (按 Episode)
    plt.figure(figsize=(12, 8))
    for method in rl_methods:
        variances = training_data[method].get("episode_variances")
        if not variances: continue

        window_size = min(10, len(variances) // 10) if len(variances) > 10 else 1
        smoothed_variances = np.convolve(variances, np.ones(window_size) / window_size, mode='valid')
        x_axis = np.arange(len(smoothed_variances)) + (window_size - 1)

        plt.plot(x_axis, smoothed_variances, label=method.upper())
        
    plt.title(f"Episodic Partition Weight Variance Curve - {graph_name}")
    plt.xlabel("Episode")
    plt.ylabel("Weight Variance")
    plt.legend()    
    plt.grid(True)
    plt.savefig(f"{results_dir}/plots/{graph_name}_episodic_variance.png")
    plt.savefig(f"{results_dir}/plots/{graph_name}_episodic_variance.svg", format='svg')
    plt.close()


def plot_avg_episode_curves_with_std(graph_name, training_data, results_dir):
    """绘制按 episode 记录的、带有标准差阴影的平均曲线"""
    # 只处理有 episodic 数据的指标
    for data_type in ["rewards", "variances"]:
        plt.figure(figsize=(12, 8))
        
        # 找出有 episodic 数据的算法
        # 修复: 从 training_data[method] 获取 episodic 数据
        methods_with_data = [m for m in training_data.keys() if training_data.get(m, {}).get(f"episode_{data_type}")]
        if not methods_with_data:
            plt.close()
            continue
        
        for method in methods_with_data:
            # 使用 episodic 数据
            data = np.array(training_data[method][f"episode_{data_type}"])
            if data.size == 0: continue

            # 使用移动窗口计算平均和标准差
            window_size = min(30, len(data) // 5) if len(data) > 30 else 1
            if window_size > 1:
                means = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
                stds = np.array([np.std(data[i:i + window_size]) for i in range(len(data) - window_size + 1)])
                
                # X轴是 episode 数
                x_axis = np.arange(len(means)) + (window_size - 1)
                
                plt.plot(x_axis, means, label=method.upper())
                plt.fill_between(x_axis, means - stds, means + stds, alpha=0.3)
            else:
                x_axis = np.arange(len(data))
                plt.plot(x_axis, data, label=method.upper())

        # 为不同数据类型设置适当的英文标题
        title_mapping = {
            "rewards": "Average Episodic Rewards",
            "variances": "Average Episodic Weight Variance"
        }
        data_type_capitalized = "Reward" if data_type == "rewards" else "Weight Variance"

        plt.title(f"{title_mapping.get(data_type, data_type.capitalize())} Curve (with Std Dev) - {graph_name}")
        plt.xlabel("Episode")
        plt.ylabel(data_type_capitalized)
        plt.legend()        
        plt.grid(True)
        # 保存为新文件名
        plt.savefig(f"{results_dir}/plots/{graph_name}_avg_episodic_{data_type}.png")
        plt.savefig(f"{results_dir}/plots/{graph_name}_avg_episodic_{data_type}.svg", format='svg')
        plt.close()


def plot_comparison(graph_name, results, results_dir):
    """绘制不同算法的比较图"""
    # 提取评估指标
    methods = list(results.keys())
    weight_variance = [results[m]["weight_variance"] for m in methods]
    edge_cut = [results[m]["normalized_cut"] for m in methods]
    execution_time = [results[m]["runtime"] for m in methods]

    # 创建子图
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 权重方差
    axs[0].bar(methods, weight_variance)
    axs[0].set_title("Weight Variance")
    axs[0].set_ylabel("Variance")

    # 归一化切边
    axs[1].bar(methods, edge_cut)
    axs[1].set_title("Normalized Edge Cut")
    axs[1].set_ylabel("Ratio")

    # 执行时间
    axs[2].bar(methods, execution_time)
    axs[2].set_title("Execution Time")
    axs[2].set_ylabel("Seconds")    
    plt.suptitle(f"Algorithm Comparison - {graph_name}")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/plots/{graph_name}_comparison.png")
    plt.savefig(f"{results_dir}/plots/{graph_name}_comparison.svg", format='svg')
    plt.close()


def main():
    # === 关键修复：设置全局随机种子确保实验可重复性 ===
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔧 全局随机种子设置为: {SEED} (确保实验可重复性)")

    # === GPU优化配置 ===
    if torch.cuda.is_available():
        # 启用CUDA优化
        torch.backends.cudnn.benchmark = True  # 针对固定输入大小优化
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32加速
        torch.backends.cudnn.allow_tf32 = True
        
        # 设置内存分配策略
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        print(f"🚀 GPU优化已启用: {torch.cuda.get_device_name(0)}")
        print(f"   - cuDNN benchmark: True")
        print(f"   - TF32 加速: True")
        print(f"   - 内存分配优化: True")
    
    # 限制 PyTorch 和底层库使用的线程数，减少后台线程空闲等待
    # 建议设置为物理核心数，例如 4 或 8，根据您的服务器调整
    num_threads = 6  # 稍微减少避免过度竞争
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    print(f"🔧 限制 PyTorch/OMP/MKL 线程数为: {num_threads}")

    # 添加这两行禁用强制同步
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_FORCE_PTX_JIT'] = '0'

    """主函数"""
    # === 修改：创建以时间戳命名的结果目录 ===
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{timestamp}"
    
    # 创建必要的目录
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    os.makedirs(f"{results_dir}/models", exist_ok=True)
    
    print(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果将保存到: {results_dir}")

    # 默认配置
    default_config = {
        "episodes": 500,
        "max_steps": 100,
        "batch_size": 32,
        "methods": ["random", "greedy", "spectral", "metis", "dqn", "gnn", "ppo", "gnn_ppo"],
        "dqn_config": {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001,
            "target_update_freq": 10
        },
        "gnn_config": {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001,
            "hidden_dim": 128,
            "target_update_freq": 10
        },
        "ppo_config": {
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "ppo_epochs": 4,
            "batch_size": 64,
            "clip_ratio": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5
        },
        "gnn_ppo_config": {
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "ppo_epochs": 4,
            "batch_size": 64,
            "clip_ratio": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "hidden_dim": 128,
            "gnn_layers": 2
        }
    }

    # 如果配置文件不存在，创建默认配置
    if not os.path.exists("configs/default.json"):
        with open("configs/default.json", "w") as f:
            json.dump(default_config, f, indent=4)

    # 加载配置
    config = load_config("configs/default.json")

    # === 修改：优先尝试加载真实图，否则使用测试图 ===
    real_graph_path = "ctu_airspace_graph_1900_2000_kmeans.graphml"
    
    if os.path.exists(real_graph_path):
        print(f"🔄 使用真实空域图: {real_graph_path}")
        try:
            graph = nx.read_graphml(real_graph_path)
            
            # 重新编号节点确保连续性
            node_mapping = {old_node: i for i, old_node in enumerate(graph.nodes())}
            graph = nx.relabel_nodes(graph, node_mapping)
            
            # 确保权重为数值类型
            for node in graph.nodes():
                if 'weight' in graph.nodes[node]:
                    graph.nodes[node]['weight'] = float(graph.nodes[node]['weight'])
                else:
                    graph.nodes[node]['weight'] = 1.0
            
            # 如果没有边，添加简单的连接
            if graph.number_of_edges() == 0:
                print("⚠️  图没有边，添加基本连接...")
                nodes = list(graph.nodes())
                for i in range(len(nodes) - 1):
                    graph.add_edge(nodes[i], nodes[i + 1])
                # 添加一些随机连接
                for _ in range(min(50, len(nodes) * 2)):
                    u, v = random.choice(nodes), random.choice(nodes)
                    if u != v:
                        graph.add_edge(u, v)
            
            num_partitions = 3 if graph.number_of_nodes() > 15 else 2
            graph_name = f"real_airspace_{graph.number_of_nodes()}nodes"
            
            print(f"✅ 真实图加载成功: {graph.number_of_nodes()}节点, {graph.number_of_edges()}边, {num_partitions}分区")
            
        except Exception as e:
            print(f"❌ 真实图加载失败: {e}")
            print("🔄 回退到测试图...")
            graph = create_test_graph(num_nodes=10, seed=42)
            num_partitions = 2
            graph_name = "test_graph_10"
    else:
        print(f"🔄 真实图文件不存在，使用测试图")
        graph = create_test_graph(num_nodes=10, seed=42)
        num_partitions = 2
        graph_name = "test_graph_10"

    print("开始图划分实验...")
    df = run_experiment(graph_name, graph, num_partitions, config, results_dir)

    # === 新增：创建训练信息记录文件 ===
    training_info = {
        "训练开始时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "训练完成时间": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "图节点数": len(graph.nodes()),
        "图边数": len(graph.edges()),
        "分区数": num_partitions,
        "训练配置": config,
        "最终结果": df.to_dict('records')
    }
    
    import json
    with open(f"{results_dir}/training_info.json", "w", encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    # 创建简洁的README文件
    readme_content = f"""# 图划分实验结果

## 训练信息
- **开始时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **图规模**: {len(graph.nodes())} 个节点, {len(graph.edges())} 条边
- **分区数**: {num_partitions}
- **训练方法**: {', '.join(config.get('methods', []))}

## 文件说明
- `plots/`: 包含所有训练曲线和比较图表
- `models/`: 包含训练好的模型文件
- `*.csv`: 实验结果数据
- `training_info.json`: 详细的训练配置和结果

## 最佳结果预览
{df.to_string(index=False)}
"""
    
    with open(f"{results_dir}/README.md", "w", encoding='utf-8') as f:
        f.write(readme_content)

    print(f"\n实验完成！结果已保存到 {results_dir} 目录")
    print(f"查看 {results_dir}/README.md 了解详细信息")
    print(df)



if __name__ == "__main__":
    main()