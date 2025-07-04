import optuna
import os
import json
import networkx as nx
import numpy as np

# 导入您项目中的核心函数
from run_experiments import train_gnn_ppo_agent, load_graph_from_file
from metrics import evaluate_partition

def objective(trial):
    """
    这是Optuna的目标函数，Optuna会不断调用它来寻找最优解。
    每一次调用，Optuna都会从我们定义的搜索空间中取一组新的超参数。
    """
    print(f"\n===== Trial #{trial.number} starting... =====")

    # 1. 定义超参数的搜索空间
    #    我们告诉Optuna可以在什么范围内调整参数
    config = {
        "episodes": 500,  # 每次试验的训练轮数可以少一点，以加快速度
        "max_steps": 100,
        "gnn_ppo_config": {
            # === 奖励函数权重 ===
            "potential_weights": {
                "variance": trial.suggest_float("variance_weight", 0.1, 20.0, log=True),
                "edge_cut": trial.suggest_float("edge_cut_weight", 0.1, 10.0, log=True),
                "modularity": trial.suggest_float("modularity_weight", 0.1, 10.0, log=True),
            },
            # === GNN-PPO Agent自身参数 ===
            "entropy_coef": trial.suggest_float("entropy_coef", 1e-3, 0.1, log=True),
            "clip_ratio": trial.suggest_float("clip_ratio", 0.1, 0.3),
            "hidden_dim": 2048, # 保持不变
            "num_partitions": 3 # 保持不变, 或根据图调整
        }
    }

    # 2. 运行一次训练
    #    我们复用您已有的训练函数
    try:
        partition, _, _, _, _, _ = train_gnn_ppo_agent(
            graph,
            num_partitions,
            config,
            results_dir=f"results/tuning_trials/trial_{trial.number}"
        )
    except Exception as e:
        print(f"Trial #{trial.number} failed with an exception: {e}")
        # 如果训练中途出错，告诉Optuna这次试验失败了
        raise optuna.exceptions.TrialPruned()


    # 3. 评估结果并计算一个最终"分数"
    #    这个分数是用来告诉Optuna这次调参的效果有多好
    if partition is None:
        print(f"Trial #{trial.number} did not produce a valid partition.")
        # 如果没有有效的划分，也视为失败
        raise optuna.exceptions.TrialPruned()

    eval_results = evaluate_partition(graph, partition, num_partitions, print_results=False)

    # === 定义我们的优化目标 ===
    # 我们希望 方差(variance) 和 归一化切边(normalized_cut) 越小越好。
    # 我们给它们分配一个固定的重要性权重，这里我们认为它们同等重要。
    # 注意：这个权重和奖励函数里的权重是两个概念。
    score = (1.0 * eval_results["weight_variance"]) + (10000 * eval_results["normalized_cut"])
    
    # 增加一个惩罚项，如果分区不平衡，分数会变得很差
    if eval_results["weight_imbalance"] > 2.0:
        score += 1e9 # 巨大的惩罚

    print(f"TRIAL #{trial.number} finished.")
    print(f"  - Params: {trial.params}")
    print(f"  - Results: variance={eval_results['weight_variance']:.2f}, norm_cut={eval_results['normalized_cut']:.4f}")
    print(f"  - FINAL SCORE: {score:.2f} (the lower the better)")

    return score


if __name__ == "__main__":
    # 加载图
    real_graph_path = "ctu_airspace_graph_1900_2000_kmeans.graphml"
    print(f"🔄 Loading graph for tuning: {real_graph_path}")
    graph = load_graph_from_file(real_graph_path)
    # 重新编号
    node_mapping = {old_node: i for i, old_node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, node_mapping)
    num_partitions = 3 if graph.number_of_nodes() > 15 else 2

    # 创建一个Optuna "study" 对象，它会管理整个优化过程
    # 我们设置一个名字，方便未来继续运行
    study_name = "gnn_ppo_tuning_study"
    storage_name = f"sqlite:///{study_name}.db"
    
    print(f"🚀 Starting Optuna study: {study_name}")
    print(f"Results will be stored in: {storage_name}")
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize", # 我们的目标是让score最小化
        load_if_exists=True # 如果数据库文件已存在，就从上次结束的地方继续
    )

    # 启动优化！Optuna会调用objective函数50次
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
            "learning_rate": study.best_trial.params["learning_rate"],
            "entropy_coef": study.best_trial.params["entropy_coef"],
            "clip_ratio": study.best_trial.params["clip_ratio"],
        }
    }
    
    output_path = "configs/best_params_generated.json"
    with open(output_path, "w") as f:
        json.dump(best_params_config, f, indent=4)
        
    print(f"\n✅ Best parameters saved to {output_path}")
    print("You can now use this file to configure your final experiment run.") 