#!/usr/bin/env python3
"""
GNN-PPO Actor网络诊断脚本
此脚本专注于诊断Actor网络的输出恒为0.5的问题
"""
import os
import argparse
import torch
import numpy as np
import networkx as nx
from run_experiments import create_test_graph
from new_environment import GraphPartitionEnvironment
from agent_ppo_gnn import GNNPPOAgent
from metrics import evaluate_partition
import matplotlib.pyplot as plt
import time
import logging
from datetime import datetime

def setup_logging(log_dir="logs"):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"actor_diagnostics_{timestamp}.log")
    
    # 配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(), log_file

def plot_logits_distribution(logits_history, output_dir="results/diagnostics"):
    """绘制logits分布图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制logits均值变化曲线
    plt.figure(figsize=(10, 6))
    means = [stats['mean'] for stats in logits_history]
    stds = [stats['std'] for stats in logits_history]
    episodes = range(len(logits_history))
    
    plt.plot(episodes, means, label='Logits Mean')
    plt.fill_between(episodes, 
                    [m - s for m, s in zip(means, stds)], 
                    [m + s for m, s in zip(means, stds)], 
                    alpha=0.3)
    plt.title('Actor网络Logits均值变化曲线')
    plt.xlabel('Episode')
    plt.ylabel('Logits值')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/logits_mean_curve.png")
    plt.close()
    
    # 绘制最小/最大值变化曲线
    plt.figure(figsize=(10, 6))
    mins = [stats['min'] for stats in logits_history]
    maxs = [stats['max'] for stats in logits_history]
    
    plt.plot(episodes, mins, label='Min', color='blue')
    plt.plot(episodes, maxs, label='Max', color='red')
    plt.title('Actor网络Logits最大/最小值变化曲线')
    plt.xlabel('Episode')
    plt.ylabel('Logits值')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/logits_range_curve.png")
    plt.close()

def plot_weight_evolution(weight_history, output_dir="results/diagnostics"):
    """绘制权重演化图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制权重差异变化曲线
    plt.figure(figsize=(10, 6))
    diffs = [stats['weight_diff'] for stats in weight_history]
    bias_diffs = [stats['bias_diff'] for stats in weight_history]
    episodes = range(len(weight_history))
    
    plt.plot(episodes, diffs, label='权重差异', color='blue')
    plt.plot(episodes, bias_diffs, label='偏置差异', color='red')
    plt.title('Actor网络最后一层权重和偏置差异变化曲线')
    plt.xlabel('Episode')
    plt.ylabel('平均绝对差值')
    plt.yscale('log')  # 使用对数尺度更好地显示小差异
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/weight_diff_curve.png")
    plt.close()

def run_diagnostics(episodes=100, max_steps=100, num_nodes=20, num_partitions=2):
    """运行Actor网络诊断"""
    logger, log_file = setup_logging()
    logger.info("==== 启动GNN-PPO Actor网络诊断 ====")
    logger.info(f"运行{episodes}个episodes，每个最多{max_steps}步")
    logger.info(f"图: {num_nodes}个节点, {num_partitions}个分区")
    
    # 创建输出目录
    os.makedirs("results/diagnostics", exist_ok=True)
    
    # 创建测试图
    graph = create_test_graph(num_nodes=num_nodes, seed=42)
    
    # 创建环境
    env = GraphPartitionEnvironment(
        graph,
        num_partitions,
        max_steps,
        gamma=0.99,
        potential_weights={'variance': 1.0, 'edge_cut': 1.0, 'modularity': 1.0}
    )
    
    # 创建配置，启用详细的健康检查
    config = {
        "hidden_dim": 64,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "ppo_epochs": 4,
        "batch_size": 32,
        "n_steps": 128,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_lr_scheduler": True,
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 10,
        "lr_scheduler_min": 1e-6,
        "use_reward_norm": True,
        
        # 健康检查设置 - 每个episode都检查
        "enable_health_check": True,
        "health_check_freq": 1,  
        "enable_grad_check": True,
        "enable_embedding_vis": True,
        "vis_freq": 10
    }
    
    # 初始化智能体
    agent = GNNPPOAgent(graph, num_partitions, config)
    
    # 记录诊断数据
    logits_history = []
    weight_history = []
    rewards_history = []
    
    # 训练循环
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        # 更新当前episode计数
        agent.current_episode = e
        
        # 重置健康检查状态
        agent.health_check_states = {
            'episode_start': False,
            'episode_end': False,
            'after_update': False
        }
        
        # 执行一步以获取诊断数据
        state_data = agent._state_to_pyg_data(state)
        
        with torch.no_grad():
            # 获取详细信息用于诊断
            _, _, _, logits, probs = agent.policy.select_action_and_log_prob(state_data)
            
            # 记录logits统计
            logits_stats = {
                'mean': logits.mean().item(),
                'std': logits.std().item(),
                'min': logits.min().item(),
                'max': logits.max().item()
            }
            logits_history.append(logits_stats)
            
            # 检查Actor网络最后一个线性层
            last_linear_layer = None
            for module in reversed(list(agent.policy.actor)):
                if isinstance(module, torch.nn.Linear):
                    last_linear_layer = module
                    break
                    
            if last_linear_layer is not None:
                weights = last_linear_layer.weight.detach()
                biases = last_linear_layer.bias.detach() if last_linear_layer.bias is not None else None
                
                # 如果分区数为2，计算两个分区对应的权重向量和偏置之间的差异
                if num_partitions == 2 and weights.size(0) == 2:
                    weight_diff = (weights[0] - weights[1]).abs().mean().item()
                    
                    if biases is not None:
                        bias_diff = abs(biases[0].item() - biases[1].item())
                    else:
                        bias_diff = 0.0
                        
                    # 记录权重统计
                    weight_stats = {
                        'weight_diff': weight_diff,
                        'bias_diff': bias_diff,
                        'weights': weights.cpu().numpy(),
                        'biases': biases.cpu().numpy() if biases is not None else None
                    }
                    weight_history.append(weight_stats)
        
        # 执行实际训练步骤
        for step in range(max_steps):
            action, log_prob, value = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.store_transition(reward, done)
            
            # 检查是否应该更新策略
            if agent.should_update():
                agent.update()
            
            state = next_state
            total_reward += reward
            if done:
                break
        
        rewards_history.append(total_reward)
        
        # 记录关键数据到日志
        if e % 5 == 0 or e == episodes - 1:
            logger.info(f"\nEpisode {e}:")
            logger.info(f"奖励: {total_reward:.2f}")
            logger.info(f"Logits统计: 均值={logits_history[-1]['mean']:.4f}, 标准差={logits_history[-1]['std']:.4f}")
            
            if weight_history:
                logger.info(f"权重差异: {weight_history[-1]['weight_diff']:.6f}")
                logger.info(f"偏置差异: {weight_history[-1]['bias_diff']:.6f}")
    
    # 绘制诊断图表
    plot_logits_distribution(logits_history)
    plot_weight_evolution(weight_history)
    
    # 绘制奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title('训练奖励曲线')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig("results/diagnostics/reward_curve.png")
    plt.close()
    
    # 保存原始诊断数据
    np.savez(
        "results/diagnostics/diagnostics_data.npz",
        logits_history=logits_history,
        weight_history=weight_history,
        rewards_history=rewards_history
    )
    
    logger.info(f"\n诊断结束，结果已保存到results/diagnostics/目录和{log_file}")
    return log_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行GNN-PPO Actor网络诊断')
    parser.add_argument('--episodes', type=int, default=100,
                      help='要运行的episodes数量')
    parser.add_argument('--max-steps', type=int, default=100,
                      help='每个episode的最大步数')
    parser.add_argument('--nodes', type=int, default=20,
                      help='图中的节点数量')
    parser.add_argument('--partitions', type=int, default=2,
                      help='分区数量')
    
    args = parser.parse_args()
    
    log_file = run_diagnostics(
        episodes=args.episodes, 
        max_steps=args.max_steps, 
        num_nodes=args.nodes, 
        num_partitions=args.partitions
    )
    
    print(f"诊断完成，日志已保存到: {log_file}")
