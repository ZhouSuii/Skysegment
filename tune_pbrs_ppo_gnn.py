import optuna
import json
import torch
# import networkx as nx # 不再直接使用
import numpy as np
from tqdm import tqdm
import time
import os
import traceback # 导入 traceback

from new_environment import GraphPartitionEnvironment # 使用新的环境
from agent_ppo_gnn import GNNPPOAgent # 确保导入更新后的 Agent
from metrics import evaluate_partition, calculate_partition_weights
from run_experiments import create_test_graph # Or load graph

# --- 配置 ---
NUM_NODES = 20 # 稍微增大图规模
NUM_PARTITIONS = 2
N_TRIALS = 150 # 根据时间和资源调整
TRAINING_EPISODES = 800 # PPO 通常需要更多回合
MAX_STEPS_PER_EPISODE = 50
OPTUNA_TIMEOUT = 3600 * 6 # 增加超时时间 (例如, 6小时)
RESULTS_DIR = "results"
CONFIGS_DIR = "configs"
BEST_PARAMS_FILE = os.path.join(CONFIGS_DIR, "best_pbrs_ppo_gnn_params.json") # 新文件名
OPTUNA_DB_NAME = "optuna_pbrs_ppo_gnn_study.db" # 新数据库名
OPTUNA_RESULTS_FILE = os.path.join(RESULTS_DIR, "optuna_pbrs_ppo_gnn_results.csv") # 新结果文件名
OPTUNA_PLOTS_DIR = os.path.join(RESULTS_DIR, "plots_v3")
REPORT_INTERVAL = 50 # 每隔多少回合报告一次中间目标值
IMBALANCE_PENALTY = 0.1 # 不平衡惩罚系数

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(OPTUNA_PLOTS_DIR, exist_ok=True)

# --- Optuna 目标函数 ---
def objective(trial: optuna.Trial):
    """GNN-PPO超参数调优的Optuna目标函数 (改进版)"""
    run_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"试验 {trial.number}: 开始于设备 {device}")

    # 1. 建议超参数 (包含 GNN, PPO 和 PBRS 权重)
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 5e-6, 1e-3, log=True), # PPO 对学习率更敏感
        'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 384, 512]), # GNN 隐藏维度
        'num_layers': trial.suggest_int('num_layers', 2, 5), # GNN 层数
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5), # GNN Dropout
        'gamma': trial.suggest_float('gamma', 0.97, 0.999, log=True), # 折扣因子
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99), # GAE lambda
        'clip_ratio': trial.suggest_float('clip_ratio', 0.1, 0.3), # PPO 裁剪系数
        'ppo_epochs': trial.suggest_int('ppo_epochs', 3, 15), # PPO 更新轮数
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]), # PPO minibatch 大小
        'entropy_coef': trial.suggest_float('entropy_coef', 1e-4, 0.05, log=True), # 熵系数
        'value_coef': trial.suggest_float('value_coef', 0.4, 0.6), # 值函数损失系数
        'update_frequency': trial.suggest_int('update_frequency', 1024, 4096), # 按步数更新的频率
        # (可选) Adam betas
        'adam_beta1': trial.suggest_float('adam_beta1', 0.85, 0.95),
        'adam_beta2': trial.suggest_float('adam_beta2', 0.99, 0.9999),
        # 固定参数
        'jit_compile': False, # JIT 对 PPO 帮助可能不大
        'use_cuda_streams': True, # 假设 Agent 内部会利用
        'use_tensorboard': False # 禁用 Tensorboard
    }

    # 建议 PBRS 势能函数权重
    potential_weights = {
        'variance': trial.suggest_float('pbrs_variance_weight', 0.1, 5.0, log=True),
        'edge_cut': trial.suggest_float('pbrs_edge_cut_weight', 0.1, 5.0, log=True),
        'modularity': trial.suggest_float('pbrs_modularity_weight', 0.1, 5.0, log=True)
    }
    print(f"试验 {trial.number}: 配置 = {config}, PBRS 权重 = {potential_weights}")

    agent = None
    env = None
    graph = None
    try:
        # 2. 设置环境和智能体
        graph = create_test_graph(num_nodes=NUM_NODES, seed=trial.number % 10) # 复用种子
        # 将 PBRS 权重和 gamma 传递给环境
        env = GraphPartitionEnvironment(
            graph,
            NUM_PARTITIONS,
            max_steps=MAX_STEPS_PER_EPISODE,
            gamma=config['gamma'], # 使用 PPO 的 gamma
            potential_weights=potential_weights
        )
        # 初始化 GNNPPOAgent
        agent = GNNPPOAgent(graph, NUM_PARTITIONS, config=config)

        # 3. 训练循环
        best_intermediate_objective = float('inf') # 记录最佳中间目标
        final_partition = None # 记录最佳分区

        # --- GPU 预热 (类似 DQN) ---
        if device.type == 'cuda':
            print(f"试验 {trial.number}: 预热 GPU...")
            temp_agent = GNNPPOAgent(graph, NUM_PARTITIONS, config=config)
            temp_state, _ = env.reset()
            # 收集足够的数据进行一次更新预热
            for _ in range(config['update_frequency'] + MAX_STEPS_PER_EPISODE):
                 temp_action, temp_log_prob, temp_value = temp_agent.act(temp_state)
                 temp_next_state, temp_reward, temp_done, _, _ = env.step(temp_action)
                 # --- 修改开始: 调用 store_transition 存储 reward 和 done ---
                 temp_agent.store_transition(float(temp_reward), temp_done)
                 temp_state = temp_next_state
                 if temp_done:
                     temp_state, _ = env.reset()
                 if temp_agent.should_update():
                     temp_agent.update() # 执行一次更新
                     break # 预热完成
            del temp_agent
            torch.cuda.synchronize()
            print(f"试验 {trial.number}: GPU 预热完成。")
        # --- 预热结束 ---

        pbar = tqdm(range(TRAINING_EPISODES), desc=f"Trial {trial.number}")
        total_steps = 0
        for e in pbar:
            state, _ = env.reset()
            episode_reward = 0
            done = False

            for step in range(MAX_STEPS_PER_EPISODE):
                # Agent 选择动作，获取 log_prob 和 value
                action, log_prob, value = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)

                # 存储完整经验
                agent.store_transition(float(reward), done)
                state = next_state
                episode_reward += reward
                total_steps += 1

                # 检查是否达到更新频率
                if agent.should_update():
                    loss = agent.update() # 执行 PPO 更新
                    if loss is not None: # 确保 update 返回了有效的 loss
                        pbar.set_postfix({"LastLoss": f"{loss:.4f}"}) # 显示最近的损失

                if done:
                    break

            pbar.set_postfix({"Reward": f"{episode_reward:.2f}", "Steps": total_steps})

            # --- 改进的剪枝报告 (类似 DQN) ---
            if (e + 1) % REPORT_INTERVAL == 0 or e == TRAINING_EPISODES - 1:
                current_partition = env.partition_assignment
                if current_partition is not None and len(np.unique(current_partition)) == NUM_PARTITIONS:
                    eval_results_intermediate = evaluate_partition(graph, current_partition, NUM_PARTITIONS, print_results=False)
                    intermediate_objective = eval_results_intermediate["normalized_cut"] + IMBALANCE_PENALTY * max(0, eval_results_intermediate["weight_imbalance"] - 1)

                    trial.report(intermediate_objective, e) # 报告目标值

                    if intermediate_objective < best_intermediate_objective:
                        best_intermediate_objective = intermediate_objective
                        final_partition = current_partition.copy()

                    if trial.should_prune():
                        print(f"试验 {trial.number}: 在回合 {e+1} 被剪枝 (中间目标值: {intermediate_objective:.4f})。")
                        return best_intermediate_objective if best_intermediate_objective != float('inf') else float('inf')
                else:
                    trial.report(float('inf'), e)
                    if trial.should_prune():
                         print(f"试验 {trial.number}: 在回合 {e+1} 因无效/未完成分区被剪枝。")
                         return float('inf')

        # 4. 最终评估 (使用记录的最佳分区)
        if final_partition is None:
            final_partition = env.partition_assignment # 备选方案
            if final_partition is None or len(np.unique(final_partition)) < NUM_PARTITIONS:
                 print(f"试验 {trial.number}: 未能生成有效的最终分区。")
                 return float('inf')

        if len(np.unique(final_partition)) < NUM_PARTITIONS:
             print(f"试验 {trial.number}: 最终记录的分区无效。")
             return float('inf')

        eval_results = evaluate_partition(graph, final_partition, NUM_PARTITIONS, print_results=False)
        print(f"试验 {trial.number}: 最终评估结果 = {eval_results}")

        # 计算最终目标值
        objective_value = eval_results["normalized_cut"] + IMBALANCE_PENALTY * max(0, eval_results["weight_imbalance"] - 1)
        print(f"试验 {trial.number}: 最终目标值 = {objective_value:.6f}")

        return objective_value

    except optuna.exceptions.TrialPruned:
        print(f"试验 {trial.number}: 确认剪枝退出。")
        return best_intermediate_objective if best_intermediate_objective != float('inf') else float('inf')
    except Exception as e:
        print(f"试验 {trial.number}: 失败，错误: {e}")
        traceback.print_exc()
        return float('inf')

    finally:
        # 5. 清理
        if agent: agent.print_performance_stats() # 打印 PPO Agent 性能统计
        del agent
        del env
        del graph
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"试验 {trial.number}: 完成。用时: {time.time() - run_start_time:.2f}s")


# --- 主执行 ---
if __name__ == "__main__":
    study_name = "gnn-pbrs-ppo-partitioning-optimization" # 新研究名称

    # Pruner 设置 (类似 DQN)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=TRAINING_EPISODES // REPORT_INTERVAL // 2,
        interval_steps=1
    )

    # --- 使用 SQLite 存储 ---
    storage_name = f"sqlite:///{OPTUNA_DB_NAME}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction='minimize',
        pruner=pruner
    )

    print(f"开始GNN-PPO的Optuna优化，共{N_TRIALS}次试验 (超时: {OPTUNA_TIMEOUT}s)...")
    print(f"数据库存储: {storage_name}")
    print(f"图大小: {NUM_NODES}个节点, {NUM_PARTITIONS}个分区")
    print(f"每次试验训练: {TRAINING_EPISODES}个回合, {MAX_STEPS_PER_EPISODE}步/回合")
    print(f"每 {REPORT_INTERVAL} 回合报告一次中间目标值用于剪枝")

    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, gc_after_trial=True)
    except KeyboardInterrupt:
        print("优化被手动停止。")
    except Exception as e:
        print(f"优化过程中发生错误: {e}")
        traceback.print_exc()

    # --- 结果处理 (与 DQN 脚本类似) ---
    print("\n" + "="*20 + " 优化完成! " + "="*20)
    print(f"数据库: {storage_name}")
    print(f"完成的试验数: {len(study.trials)}")

    # 查找、打印和保存最佳试验
    try:
        best_trial = study.best_trial
        print("\n最佳试验:")
        print(f"  编号: {best_trial.number}")
        print(f"  值 (目标): {best_trial.value:.6f}")
        print("  参数: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        with open(BEST_PARAMS_FILE, 'w') as f:
            json.dump(best_trial.params, f, indent=4)
        print(f"\n最佳超参数已保存到 {BEST_PARAMS_FILE}")
    except ValueError:
        print("\n警告: 没有试验成功完成或数据库为空。无法确定最佳试验。")
        # 尝试从数据库加载
        try:
            loaded_study = optuna.load_study(study_name=study_name, storage=storage_name)
            if loaded_study.best_trial:
                best_trial = loaded_study.best_trial
                print("\n从数据库加载的最佳试验:")
                # ... (打印信息的代码) ...
            else:
                best_trial = None
        except Exception as load_err:
            print(f"尝试从数据库加载最佳试验失败: {load_err}")
            best_trial = None
    except Exception as e:
        print(f"\n获取或保存最佳试验时出错: {e}")
        best_trial = None

    # 保存所有试验结果到 CSV
    try:
        results_df = study.trials_dataframe()
        optuna_results_path = os.path.join(RESULTS_DIR, f"{study_name}_results.csv")
        results_df.to_csv(optuna_results_path, index=False)
        print(f"Optuna试验结果已保存到 {optuna_results_path}")
    except Exception as e:
        print(f"无法保存试验结果数据框: {e}")

    # 可选：绘制图表 (与 DQN 脚本类似)
    if best_trial:
        try:
            import plotly
            import kaleido
            print("正在生成Optuna可视化图表...")
            history_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_history.png")
            importance_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_importance.png")
            parallel_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_parallel.png")
            slice_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_slice.png")
            contour_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_contour.png")

            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
            fig_slice = optuna.visualization.plot_slice(study)
            try:
                 param_importances = optuna.importance.get_param_importances(study)
                 if len(param_importances) >= 2:
                     top_two_params = list(param_importances.keys())[:2]
                     fig_contour = optuna.visualization.plot_contour(study, params=top_two_params)
                     fig_contour.write_image(contour_path)
                 else:
                     print("参数不足2个，无法生成等高线图。")
            except Exception as contour_err:
                 print(f"生成等高线图失败: {contour_err}")


            fig_history.write_image(history_path)
            fig_importance.write_image(importance_path)
            fig_parallel.write_image(parallel_path)
            fig_slice.write_image(slice_path)
            print(f"Optuna图表已保存到 {OPTUNA_PLOTS_DIR}")
        except ImportError:
            print("\n警告: 请安装plotly和kaleido以生成Optuna图表: pip install plotly kaleido")
        except Exception as e:
            print(f"\n无法生成Optuna图表: {e}")
            traceback.print_exc()
