#!/usr/bin/env python3
"""
SimplePPOAgentGNN 超参数调优脚本
使用Optuna进行贝叶斯优化搜索最优超参数
"""

import optuna
import torch
import numpy as np
import time
import json
import os
from datetime import datetime
from tqdm import tqdm

# 导入必要模块
from run_experiments import create_test_graph
from new_environment import GraphPartitionEnvironment
from agent_ppo_gnn_simple import SimplePPOAgentGNN
from metrics import evaluate_partition


# === 超参数搜索空间定义 ===
def get_search_space(trial: optuna.Trial):
    """
    定义SimplePPOAgentGNN的超参数搜索空间
    基于PPO和GNN的特性精心设计
    """
    return {
        # === 核心学习超参数 ===
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),  # 重要：影响长期奖励
        'clip_ratio': trial.suggest_float('clip_ratio', 0.1, 0.3),  # PPO核心参数
        'entropy_coef': trial.suggest_float('entropy_coef', 0.001, 0.1, log=True),  # 探索vs利用
        
        # === 网络架构超参数 ===
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
        'ppo_epochs': trial.suggest_int('ppo_epochs', 2, 8),  # 更新轮数
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        
        # === GAE和价值函数超参数 ===
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
        'value_coef': trial.suggest_float('value_coef', 0.3, 0.7),
        'update_frequency': trial.suggest_int('update_frequency', 4, 16),
        
        # === 正则化超参数 ===
        'memory_capacity': trial.suggest_categorical('memory_capacity', [5000, 10000, 20000]),
        
        # === 固定参数（减少搜索空间） ===
        'use_tensorboard': False,  # 提高训练速度
    }


def objective(trial: optuna.Trial):
    """
    Optuna目标函数：训练并评估SimplePPOAgentGNN
    """
    run_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n🔍 试验 {trial.number}: 开始超参数搜索")
    
    # === 1. 生成超参数配置 ===
    config = get_search_space(trial)
    print(f"📋 试验 {trial.number} 配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # === 2. 创建测试环境 ===
    # 使用固定的小图进行快速验证
    NUM_NODES = 12  # 适中大小，既能测试性能又不会太慢
    NUM_PARTITIONS = 3
    MAX_STEPS_PER_EPISODE = 50  # 减少步数加快训练
    TRAINING_EPISODES = 350  # 减少episode数量快速评估
    
    # === 优化：使用固定图确保公平比较 ===
    graph = create_test_graph(num_nodes=NUM_NODES, seed=42)  # 固定种子，所有trial使用相同图
    
    # 使用默认的势函数权重
    default_potential_weights = {'variance': 1.0, 'edge_cut': 1.0, 'modularity': 1.0}
    env = GraphPartitionEnvironment(
        graph,
        NUM_PARTITIONS,
        max_steps=MAX_STEPS_PER_EPISODE,
        gamma=config['gamma'],
        potential_weights=default_potential_weights
    )
    
    # === 3. 初始化智能体 ===
    # 计算状态和动作空间
    node_feature_dim = NUM_PARTITIONS + 2  # 分区 + 权重 + 度
    action_size = NUM_NODES * NUM_PARTITIONS
    
    agent = None
    best_objective_value = float('inf')
    final_partition = None
    
    try:
        agent = SimplePPOAgentGNN(node_feature_dim, action_size, config)
        
        # === 4. 训练循环 ===
        episode_rewards = []
        episode_variances = []
        best_reward = float('-inf')
        
        # 添加早停机制
        patience = 50
        no_improvement_count = 0
        best_avg_reward = float('-inf')
        
        pbar = tqdm(range(TRAINING_EPISODES), desc=f"Trial {trial.number}")
        for episode in pbar:
            # 重置环境为图数据格式
            graph_state, _ = env.reset(state_format='graph')
            total_reward = 0
            
            for step in range(MAX_STEPS_PER_EPISODE):
                # 智能体动作选择
                action = agent.act(graph_state)
                
                # 环境交互
                next_state, reward, done, _, _ = env.step(action)
                
                # 获取下一个状态的图数据格式
                next_graph_state = env.get_state('graph')
                
                # 存储经验
                agent.store_transition(reward, done)
                
                # 更新状态
                graph_state = next_graph_state
                total_reward += reward
                
                if done:
                    break
            
            # 智能体更新
            loss = agent.update()
            
            # 记录性能
            episode_rewards.append(total_reward)
            
            # 计算当前分区的方差
            if env.partition_assignment is not None:
                from metrics import calculate_weight_variance
                variance = calculate_weight_variance(graph, env.partition_assignment, NUM_PARTITIONS)
                episode_variances.append(variance)
            else:
                episode_variances.append(float('inf'))
            
            # 更新最佳奖励
            if total_reward > best_reward:
                best_reward = total_reward
                final_partition = env.partition_assignment.copy()
            
            # 更新进度条
            pbar.set_postfix({
                'reward': f'{total_reward:.2f}',
                'best': f'{best_reward:.2f}',
                'variance': f'{episode_variances[-1]:.3f}',
                'loss': f'{loss:.4f}' if loss > 0 else '0.000'
            })
            
            # === 5. 早停检查 ===
            if episode >= 20:  # 至少训练20个episode
                recent_avg_reward = np.mean(episode_rewards[-10:])  # 最近10个episode平均奖励
                
                if recent_avg_reward > best_avg_reward:
                    best_avg_reward = recent_avg_reward
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # 早停条件
                if no_improvement_count >= patience:
                    print(f"\n⏰ 试验 {trial.number}: 早停于第 {episode} episode (无改善 {no_improvement_count} 次)")
                    break
            
            # === 6. 中间剪枝 ===
            if episode >= 50 and episode % 25 == 0:
                intermediate_avg_reward = np.mean(episode_rewards[-25:])
                trial.report(intermediate_avg_reward, episode)
                
                if trial.should_prune():
                    print(f"\n✂️ 试验 {trial.number}: 被剪枝于第 {episode} episode")
                    raise optuna.exceptions.TrialPruned()
        
        # === 7. 最终评估 ===
        if final_partition is None:
            final_partition = env.partition_assignment
        
        # 处理无效分区
        if final_partition is None or len(np.unique(final_partition)) < NUM_PARTITIONS:
            print(f"❌ 试验 {trial.number}: 生成了无效分区")
            return float('inf')
        
        # 评估最终分区质量
        eval_results = evaluate_partition(graph, final_partition, NUM_PARTITIONS, print_results=False)
        
        # === 8. 目标函数设计 ===
        # 综合考虑多个指标，权重可以根据需要调整
        normalized_cut = eval_results["normalized_cut"]
        weight_imbalance = eval_results["weight_imbalance"]
        weight_variance = eval_results["weight_variance"]
        
        # 复合目标函数（越小越好）
        objective_value = (
            0.5 * normalized_cut +           # 主要目标：最小化切边
            0.3 * (weight_imbalance - 1) +   # 平衡性惩罚
            0.2 * (weight_variance / (NUM_NODES * 5))  # 方差惩罚（归一化）
        )
        
        # 奖励稳定性（负方差奖励）
        if len(episode_rewards) > 10:
            reward_stability = 1.0 / (1.0 + np.std(episode_rewards[-20:]))
            objective_value -= 0.1 * reward_stability  # 奖励稳定的试验
        
        print(f"\n📊 试验 {trial.number} 结果:")
        print(f"   归一化切边: {normalized_cut:.4f}")
        print(f"   权重不平衡: {weight_imbalance:.4f}")
        print(f"   权重方差: {weight_variance:.4f}")
        print(f"   综合目标值: {objective_value:.6f}")
        print(f"   训练时长: {time.time() - run_start_time:.2f}秒")
        
        return objective_value
        
    except Exception as e:
        print(f"💥 试验 {trial.number} 异常: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')
        
    finally:
        # === 9. 清理资源 ===
        if agent is not None:
            del agent
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"🧹 试验 {trial.number}: 资源清理完成")


def run_hyperparameter_optimization(n_trials=100, n_jobs=1):
    """
    运行超参数优化
    """
    print("🚀 开始SimplePPOAgentGNN超参数优化")
    print(f"📊 计划试验次数: {n_trials}")
    print(f"⚡ 并行任务数: {n_jobs}")
    
    # === 创建Optuna研究（支持持久化存储） ===
    study_name = "simple_ppo_gnn_optimization"  # 固定名称，支持跨session累积
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # SQLite数据库存储，支持多次运行累积结果
    storage_name = "sqlite:///optuna_simple_ppo_gnn_study.db"
    
    # 使用TPE采样器和MedianPruner剪枝器
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,       # 持久化存储
        load_if_exists=True,        # 如果已存在则加载之前的结果
        direction='minimize',       # 最小化目标函数
        sampler=optuna.samplers.TPESampler(seed=42),  # TPE采样器
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,    # 前10个试验不剪枝
            n_warmup_steps=25,      # 25个episode后开始剪枝
            interval_steps=25       # 每25个episode检查一次
        )
    )
    
    print(f"🔬 创建研究: {study_name}")
    print(f"💾 数据库存储: {storage_name}")
    
    # === 显示之前运行的结果 ===
    if len(study.trials) > 0:
        print(f"📚 已有试验数: {len(study.trials)}")
        print(f"🏆 当前最佳目标值: {study.best_value:.6f}")
        print(f"🔧 当前最佳参数预览:")
        for key, value in list(study.best_params.items())[:3]:  # 只显示前3个参数
            print(f"   {key}: {value}")
        print(f"   ... (共{len(study.best_params)}个参数)")
    else:
        print("🆕 这是全新的研究，没有历史数据")
    
    # === 运行优化 ===
    start_time = time.time()
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        print("⏸️ 用户中断优化过程")
    
    total_time = time.time() - start_time
    
    # === 结果分析 ===
    print(f"\n🎉 优化完成！总用时: {total_time/60:.1f} 分钟")
    print(f"✅ 完成试验数: {len(study.trials)}")
    print(f"🏆 最佳试验: {study.best_trial.number}")
    print(f"🎯 最佳目标值: {study.best_value:.6f}")
    
    print(f"\n🔧 最佳超参数:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # === 保存结果 ===
    results_dir = f"optimization_results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存最佳参数
    best_params_file = f"{results_dir}/best_simple_ppo_gnn_params.json"
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"💾 最佳参数已保存到: {best_params_file}")
    
    # 保存详细研究结果
    study_file = f"{results_dir}/study_simple_ppo_gnn.pkl"
    try:
        import pickle
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        print(f"💾 研究数据已保存到: {study_file}")
    except Exception as e:
        print(f"⚠️ 保存研究数据失败: {e}")
        # 保存为JSON格式作为备选
        study_json_file = f"{results_dir}/study_trials.json"
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state)
            }
            trials_data.append(trial_data)
        
        with open(study_json_file, 'w') as f:
            json.dump(trials_data, f, indent=2)
        print(f"💾 试验数据已保存到: {study_json_file}")
    
    # === 可视化分析（修复中文乱码） ===
    try:
        import matplotlib.pyplot as plt
        
        # 设置英文字体，避免中文乱码
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 参数重要性图
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title("SimplePPOAgentGNN Hyperparameter Importance", fontsize=12, pad=20)
        plt.xlabel("Importance", fontsize=10)
        plt.ylabel("Hyperparameter", fontsize=10)
        plt.tight_layout(pad=2.0)  # 增加边距避免重叠
        plt.savefig(f"{results_dir}/param_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Parameter importance plot saved")
        
        # 优化历史图
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title("SimplePPOAgentGNN Optimization History", fontsize=12, pad=20)
        plt.xlabel("Trial Number", fontsize=10)
        plt.ylabel("Objective Value", fontsize=10)
        plt.tight_layout(pad=2.0)
        plt.savefig(f"{results_dir}/optimization_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Optimization history plot saved")
        
        # 参数关系图（如果trial数>=10）
        if len(study.trials) >= 10:
            try:
                fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
                plt.title("Hyperparameter Parallel Coordinates", fontsize=12, pad=20)
                plt.tight_layout(pad=2.0)
                plt.savefig(f"{results_dir}/parallel_coordinates.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📊 Parallel coordinates plot saved")
            except Exception as e:
                print(f"⚠️ 跳过平行坐标图: {e}")
        
        plt.close('all')  # 确保关闭所有图形
        
    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
    
    return study


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SimplePPOAgentGNN超参数优化")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="试验次数 (默认: 50)")
    parser.add_argument("--n-jobs", type=int, default=2,
                       help="并行任务数 (默认: 2, 推荐1-3)")
    parser.add_argument("--quick-test", action="store_true",
                       help="快速测试模式 (10个试验)")
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("🧪 快速测试模式")
        n_trials = 10
        n_jobs = 1
    else:
        n_trials = args.n_trials
        n_jobs = args.n_jobs
    
    # 设置随机种子确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("🔧 环境检查:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 运行优化
    study = run_hyperparameter_optimization(n_trials=n_trials, n_jobs=n_jobs)
    
    print("\n🎊 优化任务完成！")
    return study


if __name__ == "__main__":
    main() 