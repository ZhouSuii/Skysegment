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
from agent_ppo_gnn import GNNPPOAgent
from metrics import calculate_weight_variance, calculate_partition_weights

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


def train_dqn_agent(graph, num_partitions, config):
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

        # 进行经验回放学习
        if memory_size >= batch_size:
            loss = agent.replay(batch_size)  # 获取返回的损失值
            loss_history.append(loss)  # 记录损失
        else:
            loss_history.append(0.0)  # 如果没有学习，记录0损失

        # 计算并记录当前分区的方差
        if env.partition_assignment is not None:
            from metrics import calculate_weight_variance
            variance = calculate_weight_variance(graph, env.partition_assignment, num_partitions)
            variance_history.append(variance)  # 记录方差
        else:
            variance_history.append(0.0)  # 默认方差为0

        # 记录奖励
        rewards_history.append(total_reward)

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
            os.makedirs("results/models", exist_ok=True)
            agent.save_model(f"results/models/dqn_model_{len(graph.nodes())}nodes_{num_partitions}parts_temp.pt")

    # 保存最终模型
    os.makedirs("results/models", exist_ok=True)
    agent.save_model(f"results/models/dqn_model_{len(graph.nodes())}nodes_{num_partitions}parts.pt")

    return best_partition, rewards_history, loss_history, variance_history


def train_gnn_agent(graph, num_partitions, config):
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

        # 检查内存池大小 - 使用memory_counter而不是len(agent.memory)
        memory_size = min(agent.memory_counter, agent.memory_capacity)

        # 进行经验回放学习
        if memory_size >= batch_size:
            loss = agent.replay()  # 获取返回的损失值
            loss_history.append(loss)  # 记录损失
        else:
            loss_history.append(0.0)  # 如果没有学习，记录0损失

        # 计算并记录当前分区的方差
        if env.partition_assignment is not None:
            from metrics import calculate_weight_variance
            variance = calculate_weight_variance(graph, env.partition_assignment, num_partitions)
            variance_history.append(variance)  # 记录方差
        else:
            variance_history.append(0.0)  # 默认方差为0

        # 记录奖励
        rewards_history.append(total_reward)

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

        # 每10个episode保存一次模型
        if (e + 1) % 50 == 0:
            os.makedirs("results/models", exist_ok=True)
            agent.save_model(f"results/models/gnn_model_{len(graph.nodes())}nodes_{num_partitions}parts_temp.pt")

    # 保存最终模型
    os.makedirs("results/models", exist_ok=True)
    agent.save_model(f"results/models/gnn_model_{len(graph.nodes())}nodes_{num_partitions}parts.pt")

    return best_partition, rewards_history, loss_history, variance_history

# 添加训练PPO智能体的函数
def train_ppo_agent(graph, num_partitions, config):
    """训练PPO智能体"""
    # 获取配置参数
    episodes = config.get("episodes", 1000)
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
    loss_history = []  # 新增：记录损失
    variance_history = []  # 新增：记录方差

    # 训练循环
    progress_bar = tqdm(range(episodes), desc="训练PPO")
    for e in progress_bar:
        state, _ = env.reset()
        total_reward = 0
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

        # 更新策略
        loss = agent.update()
        rewards_history.append(total_reward)
        loss_history.append(loss)

        # 计算当前分区权重方差
        partition_weights = calculate_partition_weights(graph, env.partition_assignment, num_partitions)
        weight_variance = np.var(partition_weights)
        variance_history.append(weight_variance)

        # 记录额外指标
        if agent.logger is not None:
            metrics = {
                "performance/weight_variance": weight_variance,
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
    os.makedirs("results/models", exist_ok=True)
    agent.save_model(f"results/models/ppo_model_{len(graph.nodes())}nodes_{num_partitions}parts.pt")

    return best_partition, rewards_history, loss_history, variance_history


# 添加训练GNN-PPO智能体的函数
def train_gnn_ppo_agent(graph, num_partitions, config):
    """训练GNN-PPO智能体"""
    # 获取配置参数
    episodes = config.get("episodes", 500)
    max_steps = config.get("max_steps", 100)
    gnn_ppo_config = config.get("gnn_ppo_config", {}) # 获取GNN-PPO配置
    
    # 启用GNN健康检查功能
    gnn_ppo_config['enable_health_check'] = config.get("enable_health_check", True)
    gnn_ppo_config['health_check_freq'] = config.get("health_check_freq", 10)
    gnn_ppo_config['enable_grad_check'] = config.get("enable_grad_check", True)
    gnn_ppo_config['enable_embedding_vis'] = config.get("enable_embedding_vis", True)
    gnn_ppo_config['vis_freq'] = config.get("vis_freq", 50)

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

    # 初始化GNN-PPO代理
    agent = GNNPPOAgent(graph, num_partitions, gnn_ppo_config)
    
    # 创建结果目录
    os.makedirs("results/embeddings", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    best_reward = float('-inf')
    best_partition = None
    rewards_history = []
    loss_history = []
    variance_history = []
    
    # 训练循环    
    progress_bar = tqdm(range(episodes), desc="训练GNN-PPO")
    start_time = time.time()
    for e in progress_bar:
        state, _ = env.reset()
        total_reward = 0
        
        # 更新当前episode计数
        agent.current_episode = e
        
        # 重置健康检查状态
        if hasattr(agent, 'health_check_states'):
            agent.health_check_states = {
                'episode_start': False,
                'episode_end': False,
                'after_update': False
            }
        
        for step in range(max_steps):
            actual_action, log_prob, value = agent.act(state) # 解包元组，获取实际动作
            next_state, reward, done, _, _ = env.step(actual_action) # 将实际动作整数传递给 env.step

            agent.store_transition(reward, done)
              # 检查是否应该更新策略 - 重要更改：每步都检查，不再等到episode结束
            if agent.should_update():
                step_loss = agent.update()
                # 记录当前更新的loss
                if len(loss_history) == e:  # 确保本episode只添加一次loss
                    loss_history.append(step_loss)
                else:
                    # 如果已有loss，则取平均
                    loss_history[e] = (loss_history[e] + step_loss) / 2.0
            
            state = next_state
            total_reward += reward
            if done:
                break

        # 如果这个episode还没记录loss，说明一直没有进行更新
        if len(loss_history) <= e:
            loss_history.append(0.0)
            
        rewards_history.append(total_reward)

        # 计算当前分区权重方差
        partition_weights = calculate_partition_weights(graph, env.partition_assignment, num_partitions)
        weight_variance = np.var(partition_weights)
        variance_history.append(weight_variance)
        
        # 在episode结束时执行健康检查 (如果配置允许)
        if hasattr(agent, 'perform_health_check') and hasattr(agent, 'health_check_states'):
            # 获取最终状态数据用于健康检查
            final_state_data = agent._state_to_pyg_data(state)
            agent.perform_health_check(final_state_data, 'episode_end')
        
        # 如果开启了健康检查并且达到了检查频率，打印摘要指标
        if agent.enable_health_check and e % agent.health_check_freq == 0:
            # 记录当前嵌入统计信息到控制台
            print(f"\n[Episode {e}] GNN-PPO性能摘要:")
            print(f"奖励: {total_reward:.2f}, 最佳奖励: {best_reward:.2f}")
            print(f"权重方差: {weight_variance:.2f}")
            print(f"损失: {loss_history[-1] if loss_history else 0:.4f}")
            
            # 如果支持TensorBoard且启用了，将统计信息记录到TensorBoard
            if agent.logger is not None:
                agent.logger.log_scalar("health/weight_variance", weight_variance, e)
        
        # 每50个episodes保存一次模型快照
        if e > 0 and e % 50 == 0:
            snapshot_path = f"results/models/gnn_ppo_snapshot_ep{e}.pt"
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

    # 打印最终性能统计
    agent.print_performance_stats()

    # 保存模型
    agent.save_model(f"results/models/gnn_ppo_model_{len(graph.nodes())}nodes_{num_partitions}parts.pt")

    return best_partition, rewards_history, loss_history, variance_history


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
            partition, rewards, losses, variances = train_dqn_agent(graph, num_partitions, config)
            # 记录训练历史
            training_data["dqn"] = {
                "rewards": rewards,
                "loss": losses,
                "variance": variances
            }
        elif method == "gnn":
            partition, rewards, losses, variances = train_gnn_agent(graph, num_partitions, config)
            # 记录训练历史
            training_data["gnn"] = {
                "rewards": rewards,
                "loss": losses,
                "variance": variances
            }
        elif method == "ppo":
            partition, rewards, losses, variances = train_ppo_agent(graph, num_partitions, config)
            # 记录训练历史
            training_data["ppo"] = {
                "rewards": rewards,
                "loss": losses,
                "variance": variances
            }
        elif method == "gnn_ppo":
            partition, rewards, losses, variances = train_gnn_ppo_agent(graph, num_partitions, config)
            # 记录训练历史
            training_data["gnn_ppo"] = {
                "rewards": rewards,
                "loss": losses,
                "variance": variances
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
        plot_training_curves(graph_name, training_data, results_dir)

    # 绘制比较图
    plot_comparison(graph_name, results, results_dir)

    return results_df


# 添加新的绘图函数
def plot_training_curves(graph_name, training_data, results_dir):
    """绘制训练历史曲线图"""
    # 只处理有训练历史的算法
    rl_methods = [m for m in training_data.keys()]
    if not rl_methods:
        return

    # 绘制奖励曲线
    plt.figure(figsize=(12, 8))
    for method in rl_methods:
        rewards = training_data[method]["rewards"]
        # 平滑处理以便更好地可视化
        window_size = min(10, len(rewards) // 10) if len(rewards) > 10 else 1
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smoothed_rewards, label=method.upper())
    plt.title(f"Training Reward Curve - {graph_name}")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()    
    plt.grid(True)
    plt.savefig(f"{results_dir}/plots/{graph_name}_rewards_comparison.png")
    plt.savefig(f"{results_dir}/plots/{graph_name}_rewards_comparison.svg", format='svg')
    plt.close()

    # 绘制损失曲线
    plt.figure(figsize=(12, 8))
    for method in rl_methods:
        loss = training_data[method]["loss"]
        # 平滑处理以便更好地可视化
        window_size = min(10, len(loss) // 10) if len(loss) > 10 else 1
        smoothed_loss = np.convolve(loss, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smoothed_loss, label=method.upper())
    plt.title(f"Training Loss Curve - {graph_name}")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.legend()    
    plt.grid(True)
    plt.savefig(f"{results_dir}/plots/{graph_name}_loss_comparison.png")
    plt.savefig(f"{results_dir}/plots/{graph_name}_loss_comparison.svg", format='svg')
    plt.close()

    # 绘制方差曲线
    plt.figure(figsize=(12, 8))
    for method in rl_methods:
        variance = training_data[method]["variance"]
        # 平滑处理以便更好地可视化
        window_size = min(10, len(variance) // 10) if len(variance) > 10 else 1
        smoothed_variance = np.convolve(variance, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smoothed_variance, label=method.upper())
    plt.title(f"Partition Weight Variance Curve - {graph_name}")
    plt.xlabel("Episodes")
    plt.ylabel("Weight Variance")
    plt.legend()    
    plt.grid(True)
    plt.savefig(f"{results_dir}/plots/{graph_name}_variance_comparison.png")
    plt.savefig(f"{results_dir}/plots/{graph_name}_variance_comparison.svg", format='svg')
    plt.close()

    # 绘制带有标准差阴影的平均曲线
    plot_avg_curves_with_std(graph_name, training_data, results_dir)


def plot_avg_curves_with_std(graph_name, training_data, results_dir):
    """绘制带有标准差阴影的平均曲线"""
    # 对每种数据类型分别绘图
    for data_type in ["rewards", "loss", "variance"]:
        plt.figure(figsize=(12, 8))

        # 找出最小长度以便对齐数据
        min_length = min(len(training_data[method][data_type]) for method in training_data.keys())

        for method in training_data.keys():
            data = np.array(training_data[method][data_type][:min_length])

            # 如果有多次运行结果，计算平均和标准差
            # 这里假设我们只有单次运行，所以模拟多次运行通过移动窗口
            window_size = min(30, len(data) // 5) if len(data) > 30 else 1
            if window_size > 1:
                # 使用移动窗口计算平均和标准差
                means = np.convolve(data, np.ones(window_size) / window_size, mode='valid')

                # 计算移动标准差
                stds = []
                for i in range(len(data) - window_size + 1):
                    stds.append(np.std(data[i:i + window_size]))
                stds = np.array(stds)

                x = np.arange(len(means))
                plt.plot(x, means, label=method.upper())
                plt.fill_between(x, means - stds, means + stds, alpha=0.3)
            else:
                # 数据太少，直接绘制
                plt.plot(data, label=method.upper())

        # 为不同数据类型设置适当的英文标题
        title_mapping = {
            "rewards": "Average Rewards",
            "loss": "Average Loss",
            "variance": "Average Weight Variance"
        }
        plt.title(f"{title_mapping.get(data_type, data_type.capitalize())} Curve (with Std Dev) - {graph_name}")
        plt.xlabel("Episodes")
        plt.ylabel(data_type.capitalize())
        plt.legend()        
        plt.grid(True)
        plt.savefig(f"{results_dir}/plots/{graph_name}_avg_{data_type}.png")
        plt.savefig(f"{results_dir}/plots/{graph_name}_avg_{data_type}.svg", format='svg')
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
    # 限制 PyTorch 和底层库使用的线程数，减少后台线程空闲等待
    # 建议设置为物理核心数，例如 4 或 8，根据您的服务器调整
    num_threads = 8
    torch.set_num_threads(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    print(f"限制 PyTorch/OMP/MKL 线程数为: {num_threads}")

    # 添加这两行禁用强制同步
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_FORCE_PTX_JIT'] = '0'

    """主函数"""
    # 创建必要的目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    # 默认配置
    default_config = {
        "episodes": 500,
        "max_steps": 100,
        "batch_size": 32,
        "methods": ["random", "greedy", "spectral", "metis", "dqn", "gnn"],
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
        }
    }

    # 如果配置文件不存在，创建默认配置
    if not os.path.exists("configs/default.json"):
        with open("configs/default.json", "w") as f:
            json.dump(default_config, f, indent=4)

    # 加载配置
    config = load_config("configs/default.json")

    # 创建测试图
    graph = create_test_graph(num_nodes=10, seed=42)
    num_partitions = 2

    print("开始图划分实验...")
    df = run_experiment("test_graph_10", graph, num_partitions, config)

    print("\n实验完成！结果已保存到results目录")
    print(df)


def run_quick_health_check(episodes=50, max_steps=100, num_nodes=20, num_partitions=2):
    """运行一个简短的训练循环，专注于GNN-PPO的健康检查"""
    print("==== 启动GNN-PPO健康检查模式 ====")
    print(f"运行{episodes}个episodes，每个最多{max_steps}步")
    print(f"图: {num_nodes}个节点, {num_partitions}个分区")
    
    # 创建测试图
    graph = create_test_graph(num_nodes=num_nodes, seed=42)
    # 使用传入的分区数量
    
    # 创建配置
    config = {
        "episodes": episodes,
        "max_steps": max_steps,
        "gnn_ppo_config": {
            "hidden_dim": 32,  # 减小隐藏维度以加快运行速度
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.1,
            "ppo_epochs": 3,
            "batch_size": 32,
            "n_steps": 128,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.3,  # 保持严格的梯度裁剪
        },
        # 健康检查设置
        "enable_health_check": True,
        "health_check_freq": 5,  # 每5个episode检查一次
        "enable_grad_check": True,
        "enable_embedding_vis": True,
        "vis_freq": 10,  # 每10个episode可视化一次
    }
    
    # 只运行GNN-PPO
    print("\n训练GNN-PPO智能体...")
    partition, rewards, losses, variances = train_gnn_ppo_agent(graph, num_partitions, config)
    
    # 评估分区质量
    print("\n评估最终分区质量...")
    eval_results = evaluate_partition(graph, partition, num_partitions)
    
    print("\n==== 健康检查结果 ====")
    print("分区质量评估:")
    for metric, value in eval_results.items():
        print(f"- {metric}: {value}")
    
    # 画出训练曲线
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title("GNN-PPO Reward Curve - Health Check")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title("GNN-PPO Loss Curve - Health Check")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(variances)
    plt.title("Partition Weight Variance - Health Check")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/plots/gnn_ppo_health_check.png")
    print("\n训练曲线已保存到 results/plots/gnn_ppo_health_check.png")
    
    return eval_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行图分区实验')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'health-check'],
                      help='运行模式: full=完整实验, health-check=GNN-PPO健康检查')
    parser.add_argument('--episodes', type=int, default=50,
                      help='健康检查模式的episodes数量')
    parser.add_argument('--max-steps', type=int, default=100,
                      help='每个episode的最大步数')
    parser.add_argument('--nodes', type=int, default=20,
                      help='图中的节点数量')
    parser.add_argument('--partitions', type=int, default=2,
                      help='分区数量')
    
    args = parser.parse_args()
    
    if args.mode == 'health-check':
        run_quick_health_check(
            episodes=args.episodes, 
            max_steps=args.max_steps,
            num_nodes=args.nodes,
            num_partitions=args.partitions
        )
    else:
        main()