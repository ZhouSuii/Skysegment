import optuna
import os
import json
import networkx as nx
import numpy as np

# 导入您项目中的核心函数
# 注意：我们将直接在objective函数中复用训练逻辑，而不是直接调用train_gnn_ppo_agent
from run_experiments import load_graph_from_file
from new_environment import GraphPartitionEnvironment
from agent_ppo_gnn_simple import SimplePPOAgentGNN
from metrics import evaluate_partition

# --- 新增辅助函数 ---
def calculate_max_variance(graph, num_partitions):
    """
    计算给定图的理论最大权重方差。
    这种情况发生在所有节点权重都集中在一个分区，而其他分区为空时。
    这个值将作为我们后续归一化的基准。
    """
    if not graph.nodes or num_partitions <= 1:
        return 1.0 # 返回1.0以避免除以零
    
    # 图中所有节点的权重总和
    total_weight = sum(d.get('weight', 1.0) for _, d in graph.nodes(data=True))
    
    # 构造最坏情况下的分区权重列表（所有权重在一个分区）
    worst_case_weights = [total_weight] + [0.0] * (num_partitions - 1)
    
    # 计算这种最坏分布下的方差
    max_var = np.var(worst_case_weights)
    
    return max_var if max_var > 0 else 1.0

# 全局变量，用于存储只计算一次的最大方差，以便objective函数可以访问
# This is a practical approach for Optuna's study.optimize interface.
max_variance = 1.0

def objective(trial):
    """
    这是Optuna的目标函数，包含了完整的训练和剪枝逻辑。
    """
    print(f"\n===== Trial #{trial.number} starting... =====")

    # 1. 定义超参数的搜索空间
    config = {
        "episodes": 500,
        "max_steps": 100,
        "gnn_ppo_config": {
            "potential_weights": {
                "variance": trial.suggest_float("variance_weight", 0.1, 20.0, log=True),
                "edge_cut": trial.suggest_float("edge_cut_weight", 0.1, 10.0, log=True),
                "modularity": trial.suggest_float("modularity_weight", 0.1, 10.0, log=True),
            },
            # === 强烈建议新增的搜索参数 ===
            "gamma": trial.suggest_float("gamma", 0.9, 0.999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 1024, 2048]),
            "entropy_coef": trial.suggest_float("entropy_coef", 1e-3, 0.1, log=True),
            "clip_ratio": trial.suggest_float("clip_ratio", 0.1, 0.3),
            "learning_rate": 0.00005,
            
            # === 新增：与 default.json 对齐的固定参数 ===
            "ppo_epochs": 6,
            "batch_size": 512,
            "value_coef": 0.43197785729901544,
            "update_frequency": 2048,
            
            "num_partitions": num_partitions
        }
    }

    # 2. 初始化环境和智能体 (这部分逻辑来自 train_gnn_ppo_agent)
    gnn_ppo_config = config["gnn_ppo_config"]
    env = GraphPartitionEnvironment(
        graph,
        num_partitions,
        config["max_steps"],
        gamma=gnn_ppo_config.get('gamma', 0.99),
        potential_weights=gnn_ppo_config["potential_weights"]
    )
    node_feature_dim = num_partitions + 2
    action_size = len(graph.nodes()) * num_partitions
    agent = SimplePPOAgentGNN(node_feature_dim, action_size, gnn_ppo_config)

    # 3. 训练循环，并在其中加入剪枝逻辑
    for e in range(config["episodes"]):
        graph_state, _ = env.reset(state_format='graph')
        
        for step in range(config["max_steps"]):
            action = agent.act(graph_state)
            _, reward, done, _, _ = env.step(action)
            next_graph_state = env.get_state('graph')
            agent.store_transition(reward, done)
            graph_state = next_graph_state
            if done:
                break
        
        agent.update()

        # === 剪枝核心逻辑 ===
        # a. 在训练中途计算一个临时评估分数
        eval_results = evaluate_partition(graph, env.partition_assignment, num_partitions, print_results=False)
        
        # === 新的、更科学的评分逻辑 ===
        # 1. 将不同量纲的指标归一化到同一个 [0, 1] 区间
        normalized_variance = eval_results["weight_variance"] / max_variance
        normalized_ncut = eval_results["normalized_cut"] / 2.0  # N-cut的理论上界是2.0

        # 2. 使用加权和计算最终分数。alpha参数代表了我们对两个目标的偏好。
        alpha = 0.5  # alpha=0.5 代表我们认为方差和切边同等重要
        intermediate_score = alpha * normalized_variance + (1 - alpha) * normalized_ncut

        # b. 向Optuna汇报当前的分数和步数
        trial.report(intermediate_score, e)

        # c. 询问Optuna是否应该剪枝
        if trial.should_prune():
            print(f"Trial #{trial.number} pruned at episode {e} with score {intermediate_score:.4f}.")
            raise optuna.exceptions.TrialPruned()

    # 4. 如果训练正常完成，返回最终分数
    final_eval_results = evaluate_partition(graph, env.partition_assignment, num_partitions, print_results=False)
    
    # 重复使用相同的科学评分逻辑
    normalized_variance = final_eval_results["weight_variance"] / max_variance
    normalized_ncut = final_eval_results["normalized_cut"] / 2.0
    alpha = 0.5
    final_score = alpha * normalized_variance + (1 - alpha) * normalized_ncut
    
    if final_eval_results["weight_imbalance"] > 2.0:
        final_score += 1e9 # 保持对严重不平衡的巨大惩罚

    print(f"TRIAL #{trial.number} finished.")
    print(f"  - Params: {trial.params}")
    print(f"  - Raw Results: variance={final_eval_results['weight_variance']:.2f}, norm_cut={final_eval_results['normalized_cut']:.4f}")
    print(f"  - Normalized Score Components: norm_var={normalized_variance:.4f}, norm_ncut={normalized_ncut:.4f}")
    print(f"  - FINAL SCORE: {final_score:.4f} (the lower the better, based on normalized metrics)")

    return final_score


if __name__ == "__main__":
    # 加载图
    real_graph_path = "ctu_airspace_graph_1900_2000_kmeans.graphml"
    print(f"🔄 Loading graph for tuning: {real_graph_path}")
    graph = load_graph_from_file(real_graph_path)
    # 重新编号
    node_mapping = {old_node: i for i, old_node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, node_mapping)
    num_partitions = 3 if graph.number_of_nodes() > 15 else 2

    # --- 新增: 在开始前，计算一次理论最大方差 ---
    max_variance = calculate_max_variance(graph, num_partitions)
    print(f"⚖️  Calculated theoretical max variance for normalization: {max_variance:.2f}")

    # 创建一个Optuna "study" 对象
    study_name = "gnn_ppo_tuning_study"
    storage_name = f"sqlite:///{study_name}.db"
    
    print(f"🚀 Starting Optuna study: {study_name}")
    print(f"Results will be stored in: {storage_name}")
    
    # === 新增：配置剪枝器 ===
    # 我们使用中位数剪枝器，它会在若干步后，比较当前试验和历史试验的中位数表现
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # 前5次试验不做剪枝，用于收集基准数据
        n_warmup_steps=100,   # 前50个episodes不做剪枝，让模型先热身
        interval_steps=10    # 每10个episodes检查一次是否要剪枝
    )
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize", 
        load_if_exists=True,
        pruner=pruner  # <-- 将剪枝器应用到study中
    )

    # 启动优化
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # 实验结束，打印最佳结果
    print("\n\n🎉🎉🎉 TUNING COMPLETE! 🎉🎉🎉")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"  - Score: {study.best_trial.value:.2f}")
    print("  - Best hyperparameters:")
    # === 修复：合并固定参数和搜索到的参数 ===
    final_params = study.best_trial.params.copy()
    # 手动加入在搜索中被固定的参数
    final_params["learning_rate"] = 0.00005
    final_params["ppo_epochs"] = 6
    final_params["batch_size"] = 512
    final_params["value_coef"] = 0.43197785729901544
    final_params["update_frequency"] = 2048

    for key, value in final_params.items():
        print(f"    - {key}: {value}")

    # === 修复：将所有相关的最佳参数保存到JSON文件 ===
    best_params_config = {
        "gnn_ppo_config": {
            "potential_weights": {
                "variance": final_params["variance_weight"],
                "edge_cut": final_params["edge_cut_weight"],
                "modularity": final_params["modularity_weight"],
            },
            "learning_rate": final_params["learning_rate"],
            "gamma": final_params["gamma"],
            "gae_lambda": final_params["gae_lambda"],
            "hidden_dim": final_params["hidden_dim"],
            "entropy_coef": final_params["entropy_coef"],
            "clip_ratio": final_params["clip_ratio"],
            "ppo_epochs": final_params["ppo_epochs"],
            "batch_size": final_params["batch_size"],
            "value_coef": final_params["value_coef"],
            "update_frequency": final_params["update_frequency"]
        }
    }
    
    output_path = "configs/best_params_generated.json"
    with open(output_path, "w") as f:
        json.dump(best_params_config, f, indent=4)
        
    print(f"\n✅ Best parameters saved to {output_path}")
    print("You can now use this file to configure your final experiment run.") 