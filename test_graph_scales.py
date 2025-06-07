#!/usr/bin/env python3
"""
多规模图测试：验证GNN在不同图规模下的优势
"""
import time
import numpy as np
import networkx as nx
from new_environment import GraphPartitionEnvironment
from agent_ppo import PPOAgent
from agent_ppo_gnn_simple import SimplePPOAgentGNN

def create_test_graphs():
    """创建不同规模的测试图"""
    graphs = {}
    
    # 小图：10节点 (当前测试)
    G_small = nx.random_geometric_graph(10, 0.5, seed=42)
    for i in range(10):
        G_small.nodes[i]['weight'] = np.random.randint(1, 10)
    graphs['small_10'] = G_small
    
    # 中图：20节点
    G_medium = nx.random_geometric_graph(20, 0.4, seed=42)
    for i in range(20):
        G_medium.nodes[i]['weight'] = np.random.randint(1, 10)
    graphs['medium_20'] = G_medium
    
    # 大图：50节点
    G_large = nx.random_geometric_graph(50, 0.3, seed=42)
    for i in range(50):
        G_large.nodes[i]['weight'] = np.random.randint(1, 10)
    graphs['large_50'] = G_large
    
    return graphs

def quick_train_test(graph, agent_type, graph_name, episodes=100):
    """快速训练测试"""
    print(f"\n🔄 测试 {graph_name} - {agent_type}")
    
    num_nodes = len(graph.nodes())
    num_partitions = 2
    max_steps = min(50, num_nodes * 2)  # 自适应步数
    
    # 创建环境
    env = GraphPartitionEnvironment(graph, num_partitions, max_steps)
    
    # 创建智能体
    if agent_type == "PPO":
        state_size = num_nodes * (num_partitions + 2)
        action_size = num_nodes * num_partitions
        agent = PPOAgent(state_size, action_size, {
            'learning_rate': 0.0003,
            'batch_size': min(32, max(8, num_nodes)),
            'ppo_epochs': 3,
            'use_tensorboard': False
        })
    else:  # SimplePPOGNN
        node_feature_dim = num_partitions + 2
        action_size = num_nodes * num_partitions
        agent = SimplePPOAgentGNN(node_feature_dim, action_size, {
            'learning_rate': 0.0001,
            'batch_size': min(32, max(8, num_nodes)),
            'ppo_epochs': 3,
            'hidden_dim': min(64, max(32, num_nodes)),
            'use_tensorboard': False
        })
    
    # 快速训练
    best_reward = float('-inf')
    final_rewards = []
    start_time = time.time()
    
    for episode in range(episodes):
        if agent_type == "PPO":
            state, _ = env.reset()
        else:
            state, _ = env.reset(state_format='graph')
        
        total_reward = 0
        for step in range(max_steps):
            action = agent.act(state)
            if agent_type == "PPO":
                next_state, reward, done, _, _ = env.step(action)
            else:
                next_state, reward, done, _, _ = env.step(action)
                next_state = env.get_state('graph')
            
            agent.store_transition(reward, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.update()
        final_rewards.append(total_reward)
        best_reward = max(best_reward, total_reward)
        
        # 每25个episode打印进度
        if (episode + 1) % 25 == 0:
            avg_reward = np.mean(final_rewards[-25:])
            print(f"  Episode {episode+1}: avg_reward={avg_reward:.1f}, best={best_reward:.1f}")
    
    training_time = time.time() - start_time
    
    # 计算最终指标
    from metrics import evaluate_partition
    final_eval = evaluate_partition(graph, env.partition_assignment, num_partitions, print_results=False)
    
    return {
        'best_reward': best_reward,
        'avg_final_reward': np.mean(final_rewards[-10:]),  # 最后10个episode平均
        'training_time': training_time,
        'weight_variance': final_eval['weight_variance'],
        'edge_cut': final_eval['edge_cut'],
        'num_nodes': num_nodes,
        'num_edges': len(graph.edges())
    }

def run_scale_comparison():
    """运行多规模比较"""
    print("🎯 开始多规模图测试...")
    
    graphs = create_test_graphs()
    results = {}
    
    for graph_name, graph in graphs.items():
        print(f"\n📊 测试图: {graph_name} ({len(graph.nodes())}节点, {len(graph.edges())}边)")
        
        # 测试原版PPO
        try:
            ppo_result = quick_train_test(graph, "PPO", graph_name, episodes=100)
            results[f"{graph_name}_PPO"] = ppo_result
        except Exception as e:
            print(f"  ❌ PPO测试失败: {e}")
            results[f"{graph_name}_PPO"] = None
        
        # 测试简化GNN-PPO
        try:
            gnn_result = quick_train_test(graph, "SimplePPOGNN", graph_name, episodes=100)
            results[f"{graph_name}_SimplePPOGNN"] = gnn_result
        except Exception as e:
            print(f"  ❌ SimplePPOGNN测试失败: {e}")
            results[f"{graph_name}_SimplePPOGNN"] = None
    
    # 分析结果
    print("\n" + "="*80)
    print("📈 多规模测试结果分析")
    print("="*80)
    
    for graph_name in ['small_10', 'medium_20', 'large_50']:
        ppo_key = f"{graph_name}_PPO"
        gnn_key = f"{graph_name}_SimplePPOGNN"
        
        if results.get(ppo_key) and results.get(gnn_key):
            ppo_res = results[ppo_key]
            gnn_res = results[gnn_key]
            
            print(f"\n🔍 {graph_name.upper()} ({ppo_res['num_nodes']}节点):")
            print(f"  权重方差:    PPO={ppo_res['weight_variance']:.1f}  vs  GNN={gnn_res['weight_variance']:.1f}")
            print(f"  边切割:      PPO={ppo_res['edge_cut']:.1f}      vs  GNN={gnn_res['edge_cut']:.1f}")
            print(f"  训练时间:    PPO={ppo_res['training_time']:.1f}s   vs  GNN={gnn_res['training_time']:.1f}s")
            print(f"  最终奖励:    PPO={ppo_res['avg_final_reward']:.1f}    vs  GNN={gnn_res['avg_final_reward']:.1f}")
            
            # 计算改进百分比
            if ppo_res['weight_variance'] > 0:
                var_improvement = (ppo_res['weight_variance'] - gnn_res['weight_variance']) / ppo_res['weight_variance'] * 100
                print(f"  方差改进:    {var_improvement:+.1f}% (负数=GNN更差)")
            
            if ppo_res['avg_final_reward'] != 0:
                reward_improvement = (gnn_res['avg_final_reward'] - ppo_res['avg_final_reward']) / abs(ppo_res['avg_final_reward']) * 100
                print(f"  奖励改进:    {reward_improvement:+.1f}% (正数=GNN更好)")
    
    return results

if __name__ == "__main__":
    results = run_scale_comparison()
    print("\n🏁 多规模测试完成！")
    print("\n💡 分析建议:")
    print("  - 如果小图上GNN劣势明显，属正常现象")
    print("  - 如果中图/大图上GNN仍无优势，需重新考虑架构")
    print("  - 关注训练时间vs性能的权衡") 