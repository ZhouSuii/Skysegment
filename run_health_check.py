#!/usr/bin/env python3
"""
GNN-PPO健康检查运行脚本
此脚本运行一个简短的训练循环，专注于监控GNN-PPO模型的运行状态
"""
import argparse
from run_experiments import run_quick_health_check

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行GNN-PPO健康检查')
    parser.add_argument('--episodes', type=int, default=50,
                      help='要运行的episodes数量')
    parser.add_argument('--max-steps', type=int, default=100,
                      help='每个episode的最大步数')
    parser.add_argument('--nodes', type=int, default=20,
                      help='图中的节点数量')
    parser.add_argument('--partitions', type=int, default=2,
                      help='分区数量')
    
    args = parser.parse_args()
    
    print(f"启动GNN-PPO健康检查: {args.episodes}个episodes，每个最多{args.max_steps}步")
    print(f"图: {args.nodes}个节点, {args.partitions}个分区")
    
    run_quick_health_check(
        episodes=args.episodes, 
        max_steps=args.max_steps, 
        num_nodes=args.nodes, 
        num_partitions=args.partitions
    )
