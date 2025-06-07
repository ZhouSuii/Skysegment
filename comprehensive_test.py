#!/usr/bin/env python3
"""
全面测试框架：比较不同GNN-PPO版本在多规模图上的表现
"""
import time
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

from new_environment import GraphPartitionEnvironment
from agent_ppo import PPOAgent
from agent_ppo_gnn_simple import SimplePPOAgentGNN
from agent_ppo_gnn_adaptive import AdaptivePPOAgentGNN
from metrics import evaluate_partition

class ComprehensiveTestFramework:
    def __init__(self, results_dir=None):
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"comprehensive_test_{timestamp}"
        else:
            self.results_dir = results_dir
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/plots", exist_ok=True)
        
        self.results = []
        
    def create_test_graphs(self):
        """创建多样化的测试图"""
        graphs = {}
        
        # 小图系列
        for nodes in [8, 10, 12, 15]:
            G = nx.random_geometric_graph(nodes, 0.5, seed=42)
            for i in range(nodes):
                G.nodes[i]['weight'] = np.random.randint(1, 10)
            graphs[f'small_{nodes}'] = G
        
        # 中图系列
        for nodes in [20, 25, 30]:
            G = nx.random_geometric_graph(nodes, 0.4, seed=42)
            for i in range(nodes):
                G.nodes[i]['weight'] = np.random.randint(1, 10)
            graphs[f'medium_{nodes}'] = G
        
        # 大图系列
        for nodes in [40, 50]:
            G = nx.random_geometric_graph(nodes, 0.3, seed=42)
            for i in range(nodes):
                G.nodes[i]['weight'] = np.random.randint(1, 10)
            graphs[f'large_{nodes}'] = G
        
        return graphs
    
    def test_single_agent(self, graph, graph_name, agent_type, episodes=100):
        """测试单个智能体"""
        print(f"\n🔍 测试 {graph_name} - {agent_type}")
        
        num_nodes = len(graph.nodes())
        num_partitions = 2
        max_steps = min(50, num_nodes * 2)
        
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
            use_graph_format = False
        elif agent_type == "SimplePPOGNN":
            node_feature_dim = num_partitions + 2
            action_size = num_nodes * num_partitions
            agent = SimplePPOAgentGNN(node_feature_dim, action_size, {
                'learning_rate': 0.0001,
                'batch_size': min(32, max(8, num_nodes)),
                'ppo_epochs': 3,
                'hidden_dim': 64,
                'use_tensorboard': False
            })
            use_graph_format = True
        elif agent_type == "AdaptivePPOGNN":
            node_feature_dim = num_partitions + 2
            action_size = num_nodes * num_partitions
            agent = AdaptivePPOAgentGNN(node_feature_dim, action_size, num_nodes, {})
            use_graph_format = True
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # 训练
        rewards_history = []
        best_reward = float('-inf')
        start_time = time.time()
        
        for episode in range(episodes):
            if use_graph_format:
                state, _ = env.reset(state_format='graph')
            else:
                state, _ = env.reset()
            
            total_reward = 0
            
            for step in range(max_steps):
                action = agent.act(state)
                
                if use_graph_format:
                    next_state, reward, done, _, _ = env.step(action)
                    next_state = env.get_state('graph')
                else:
                    next_state, reward, done, _, _ = env.step(action)
                
                agent.store_transition(reward, done)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # 更新策略
            agent.update()
            
            rewards_history.append(total_reward)
            best_reward = max(best_reward, total_reward)
            
            # 每25个episode打印进度
            if (episode + 1) % 25 == 0:
                recent_avg = np.mean(rewards_history[-25:])
                print(f"  Episode {episode+1}: avg_reward={recent_avg:.1f}, best={best_reward:.1f}")
        
        training_time = time.time() - start_time
        
        # 最终评估
        final_eval = evaluate_partition(graph, env.partition_assignment, num_partitions, print_results=False)
        
        result = {
            'graph_name': graph_name,
            'agent_type': agent_type,
            'num_nodes': num_nodes,
            'num_edges': len(graph.edges()),
            'final_weight_variance': final_eval['weight_variance'],
            'final_edge_cut': final_eval['edge_cut'],
            'best_reward': best_reward,
            'final_avg_reward': np.mean(rewards_history[-10:]),
            'training_time': training_time,
            'convergence_episode': len(rewards_history)
        }
        
        return result
    
    def run_comprehensive_test(self, episodes=100):
        """运行全面测试"""
        print("🚀 开始全面测试...")
        
        graphs = self.create_test_graphs()
        agents = ["PPO", "SimplePPOGNN", "AdaptivePPOGNN"]
        
        for graph_name, graph in graphs.items():
            print(f"\n📊 测试图: {graph_name} ({len(graph.nodes())}节点)")
            
            for agent_type in agents:
                try:
                    result = self.test_single_agent(graph, graph_name, agent_type, episodes)
                    self.results.append(result)
                except Exception as e:
                    print(f"  ❌ {agent_type} 测试失败: {e}")
        
        # 生成分析
        self._generate_analysis()
        print(f"\n🎉 测试完成！结果保存在 {self.results_dir}")
    
    def _generate_analysis(self):
        """生成分析报告"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(f"{self.results_dir}/results.csv", index=False)
        
        # 生成规模分析图
        plt.figure(figsize=(15, 10))
        
        # 权重方差 vs 节点数
        plt.subplot(2, 2, 1)
        for agent in df['agent_type'].unique():
            agent_data = df[df['agent_type'] == agent]
            plt.plot(agent_data['num_nodes'], agent_data['final_weight_variance'], 
                    marker='o', label=agent)
        plt.xlabel('Number of Nodes')
        plt.ylabel('Weight Variance')
        plt.title('Weight Variance vs Graph Size')
        plt.legend()
        plt.grid(True)
        
        # 训练时间 vs 节点数
        plt.subplot(2, 2, 2)
        for agent in df['agent_type'].unique():
            agent_data = df[df['agent_type'] == agent]
            plt.plot(agent_data['num_nodes'], agent_data['training_time'], 
                    marker='s', label=agent)
        plt.xlabel('Number of Nodes')
        plt.ylabel('Training Time (s)')
        plt.title('Training Time vs Graph Size')
        plt.legend()
        plt.grid(True)
        
        # 最终奖励 vs 节点数
        plt.subplot(2, 2, 3)
        for agent in df['agent_type'].unique():
            agent_data = df[df['agent_type'] == agent]
            plt.plot(agent_data['num_nodes'], agent_data['final_avg_reward'], 
                    marker='^', label=agent)
        plt.xlabel('Number of Nodes')
        plt.ylabel('Final Average Reward')
        plt.title('Performance vs Graph Size')
        plt.legend()
        plt.grid(True)
        
        # 边切割 vs 节点数
        plt.subplot(2, 2, 4)
        for agent in df['agent_type'].unique():
            agent_data = df[df['agent_type'] == agent]
            plt.plot(agent_data['num_nodes'], agent_data['final_edge_cut'], 
                    marker='d', label=agent)
        plt.xlabel('Number of Nodes')
        plt.ylabel('Edge Cut')
        plt.title('Edge Cut vs Graph Size')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/comprehensive_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成文字报告
        with open(f"{self.results_dir}/report.md", "w") as f:
            f.write("# 全面测试报告\n\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 性能总结\n\n")
            for agent in df['agent_type'].unique():
                agent_data = df[df['agent_type'] == agent]
                f.write(f"### {agent}\n")
                f.write(f"- 平均权重方差: {agent_data['final_weight_variance'].mean():.2f}\n")
                f.write(f"- 平均训练时间: {agent_data['training_time'].mean():.1f}s\n")
                f.write(f"- 平均最终奖励: {agent_data['final_avg_reward'].mean():.1f}\n\n")

def main():
    """主函数"""
    print("🧪 启动全面测试框架...")
    
    framework = ComprehensiveTestFramework()
    framework.run_comprehensive_test(episodes=100)

if __name__ == "__main__":
    main() 