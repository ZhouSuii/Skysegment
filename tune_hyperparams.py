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
            "entropy_coef": trial.suggest_float("entropy_coef", 1e-3, 0.1, log=True),
            "clip_ratio": trial.suggest_float("clip_ratio", 0.1, 0.3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "hidden_dim": 2048,
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
        intermediate_score = (1.0 * eval_results["weight_variance"]) + (10000 * eval_results["normalized_cut"])
        
        # b. 向Optuna汇报当前的分数和步数
        trial.report(intermediate_score, e)

        # c. 询问Optuna是否应该剪枝
        if trial.should_prune():
            print(f"Trial #{trial.number} pruned at episode {e} with score {intermediate_score:.2f}.")
            raise optuna.exceptions.TrialPruned()

    # 4. 如果训练正常完成，返回最终分数
    final_eval_results = evaluate_partition(graph, env.partition_assignment, num_partitions, print_results=False)
    final_score = (1.0 * final_eval_results["weight_variance"]) + (10000 * final_eval_results["normalized_cut"])
    
    if final_eval_results["weight_imbalance"] > 2.0:
        final_score += 1e9

    print(f"TRIAL #{trial.number} finished.")
    print(f"  - Params: {trial.params}")
    print(f"  - Results: variance={final_eval_results['weight_variance']:.2f}, norm_cut={final_eval_results['normalized_cut']:.4f}")
    print(f"  - FINAL SCORE: {final_score:.2f} (the lower the better)")

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
    study.optimize(objective, n_trials=50)

    # 实验结束，打印最佳结果
    print("\n\n🎉🎉🎉 TUNING COMPLETE! 🎉🎉🎉")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"  - Score: {study.best_trial.value:.2f}")
    print("  - Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    - {key}: {value}")

    # 将最佳参数保存到JSON文件
    best_params_config = {
        "gnn_ppo_config": {
            "potential_weights": {
                "variance": study.best_trial.params["variance_weight"],
                "edge_cut": study.best_trial.params["edge_cut_weight"],
                "modularity": study.best_trial.params["modularity_weight"],
            },
            "learning_rate": study.best_trial.params.get("learning_rate"), # .get()更安全
            "entropy_coef": study.best_trial.params["entropy_coef"],
            "clip_ratio": study.best_trial.params["clip_ratio"],
        }
    }
    
    output_path = "configs/best_params_generated.json"
    with open(output_path, "w") as f:
        json.dump(best_params_config, f, indent=4)
        
    print(f"\n✅ Best parameters saved to {output_path}")
    print("You can now use this file to configure your final experiment run.") 