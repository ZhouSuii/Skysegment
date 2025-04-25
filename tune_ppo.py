import optuna
import json
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
import time
import os

from environment import GraphPartitionEnvironment
from agent_ppo import PPOAgent
from metrics import evaluate_partition, calculate_partition_weights
from run_experiments import create_test_graph

# --- 配置 ---
NUM_NODES = 10  # 使用中等大小的图进行调优
NUM_PARTITIONS = 2
N_TRIALS = 200  # Optuna试验次数
TRAINING_EPISODES = 250  # 每次试验的训练回合数(对PPO可以稍微减少)
MAX_STEPS_PER_EPISODE = 50  # 每回合最大步数
OPTUNA_TIMEOUT = 3600  # 整个优化过程的超时时间(例如,1小时)
RESULTS_DIR = "results"
CONFIGS_DIR = "configs"
BEST_PARAMS_FILE = os.path.join(CONFIGS_DIR, "best_ppo_params.json")
OPTUNA_RESULTS_FILE = os.path.join(RESULTS_DIR, "optuna_ppo_results.csv")
OPTUNA_PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(OPTUNA_PLOTS_DIR, exist_ok=True)

# --- Optuna目标函数 ---
def objective(trial: optuna.Trial):
    """PPO超参数调优的Optuna目标函数"""
    run_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"试验 {trial.number}: 开始于设备 {device}")

    # 1. 建议超参数 - PPO特有的搜索空间
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 5e-5, 1e-3, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),  # 隐藏层维度
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),  # 折扣因子 - PPO通常需要较高的gamma
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),  # GAE lambda参数
        'clip_ratio': trial.suggest_float('clip_ratio', 0.1, 0.3),  # PPO裁剪参数
        'ppo_epochs': trial.suggest_int('ppo_epochs', 2, 10),  # 每次更新的PPO迭代次数
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),  # 批量大小
        'entropy_coef': trial.suggest_float('entropy_coef', 0.0, 0.02),  # 熵正则化系数
        'value_coef': trial.suggest_float('value_coef', 0.4, 0.6),  # 值损失系数
        'update_frequency': trial.suggest_int('update_frequency', 4, 16),  # 更新频率，优化CPU-GPU数据传输
        # 固定参数
        'use_tensorboard': False,  # 禁用TensorBoard，减少CPU开销
        'pin_memory': True  # 启用pin_memory加速GPU传输
    }
    print(f"试验 {trial.number}: 配置 = {config}")

    agent = None
    env = None
    try:
        # 2. 设置环境和智能体
        graph = create_test_graph(num_nodes=NUM_NODES, seed=trial.number)
        env = GraphPartitionEnvironment(graph, NUM_PARTITIONS, max_steps=MAX_STEPS_PER_EPISODE)

        # 计算状态和动作空间大小
        num_nodes = len(graph.nodes())
        state_size = num_nodes * (NUM_PARTITIONS + 1)  # 扁平化状态大小
        action_size = num_nodes * NUM_PARTITIONS

        agent = PPOAgent(state_size, action_size, config=config)

        # 3. 训练循环
        best_reward_trial = float('-inf')
        final_partition = None

        # 预分配数组，减少内存分配
        rewards = np.zeros(TRAINING_EPISODES)
        losses = np.zeros(TRAINING_EPISODES)
        variances = np.zeros(TRAINING_EPISODES)
        
        # 为批量处理预热GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        # 训练前预热，确保GPU处于活跃状态
        state, _ = env.reset()
        for _ in range(3):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transition(reward, done)
            state = next_state
            if done:
                state, _ = env.reset()
        # 预热更新
        agent.update()

        # 使用tqdm显示训练进度
        for e in range(TRAINING_EPISODES):
            episode_start_time = time.time()
            state, _ = env.reset()
            total_reward = 0
            done = False
            steps_in_episode = 0

            for step in range(MAX_STEPS_PER_EPISODE):
                # 使用智能体选择动作 - 直接使用经过优化的act方法
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)

                # PPO特有的存储方式 - 只存储奖励和是否结束
                agent.store_transition(reward, done)
                state = next_state
                total_reward += reward
                steps_in_episode += 1

                if done:
                    break

            # 每个回合结束后进行一次策略更新
            loss = agent.update()
            rewards[e] = total_reward
            losses[e] = loss
            
            # 计算当前分区权重方差，作为平衡度的指标
            if env.partition_assignment is not None:
                weights = calculate_partition_weights(graph, env.partition_assignment, NUM_PARTITIONS)
                variance = np.var(weights) if len(weights) > 0 else 0
            else:
                variance = 0
            variances[e] = variance

            # 更新最佳奖励
            if total_reward > best_reward_trial:
                best_reward_trial = total_reward
                final_partition = env.partition_assignment.copy() if env.partition_assignment is not None else None

            # Optuna剪枝：检查是否应提前停止试验
            trial.report(total_reward, e)
            if trial.should_prune():
                print(f"试验 {trial.number}: 在回合 {e} 被剪枝，奖励值 {total_reward}")
                raise optuna.exceptions.TrialPruned()

            if (e + 1) % 25 == 0:  # 定期打印进度
                print(f"试验 {trial.number} - 回合 {e+1}/{TRAINING_EPISODES} - 奖励: {total_reward:.2f}, 损失: {loss:.4f}, 方差: {variance:.4f}, 用时: {time.time()-episode_start_time:.2f}s")

        # 4. 评估
        if final_partition is None:
            final_partition = env.partition_assignment

        # 处理训练失败或产生无效分区的情况
        if final_partition is None or len(np.unique(final_partition)) < NUM_PARTITIONS:
            print(f"试验 {trial.number}: 生成了无效分区。返回高惩罚值。")
            return float('inf')  # 返回一个大值表示失败(因为我们是最小化目标)

        eval_results = evaluate_partition(graph, final_partition, NUM_PARTITIONS, print_results=False)
        print(f"试验 {trial.number}: 评估结果 = {eval_results}")

        # 定义目标：最小化归一化割边 + 不平衡惩罚
        objective_value = eval_results["normalized_cut"] + 0.1 * (eval_results["weight_imbalance"] - 1)
        print(f"试验 {trial.number}: 目标值 = {objective_value:.6f}")

        return objective_value

    except Exception as e:
        print(f"试验 {trial.number}: 失败，错误: {e}")
        import traceback
        traceback.print_exc()
        # 返回大值表示失败
        return float('inf')

    finally:
        # 5. 清理GPU内存
        del agent
        del env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"试验 {trial.number}: 完成。用时: {time.time() - run_start_time:.2f}s")


# --- 主执行 ---
if __name__ == "__main__":
    # 创建研究
    study_name = "ppo-partitioning-optimization"
    
    # 使用MedianPruner提前停止没有希望的试验
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=TRAINING_EPISODES // 4, interval_steps=10)
    study = optuna.create_study(direction='minimize', pruner=pruner)

    print(f"开始PPO的Optuna优化，共{N_TRIALS}次试验 (超时: {OPTUNA_TIMEOUT}s)...")
    print(f"图大小: {NUM_NODES}个节点, {NUM_PARTITIONS}个分区")
    print(f"每次试验训练: {TRAINING_EPISODES}个回合, {MAX_STEPS_PER_EPISODE}步/回合")

    try:
        # 使用多进程可以进一步提高效率，但需要注意内存占用
        study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, gc_after_trial=True)
    except KeyboardInterrupt:
        print("优化被手动停止。")
    except Exception as e:
        print(f"优化过程中发生错误: {e}")

    # --- 结果 ---
    print("\n" + "="*20 + " 优化完成! " + "="*20)
    print(f"完成的试验数: {len(study.trials)}")

    # 查找并打印最佳试验
    try:
        best_trial = study.best_trial
        print("\n最佳试验:")
        print(f"  值 (目标): {best_trial.value:.6f}")
        print("  参数: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # 保存最佳超参数
        with open(BEST_PARAMS_FILE, 'w') as f:
            json.dump(best_trial.params, f, indent=4)
        print(f"\n最佳超参数已保存到 {BEST_PARAMS_FILE}")

    except ValueError:
        print("\n没有试验成功完成。无法确定最佳试验。")
        best_trial = None

    # 保存所有试验结果
    try:
        results_df = study.trials_dataframe()
        results_df.to_csv(OPTUNA_RESULTS_FILE, index=False)
        print(f"Optuna试验结果已保存到 {OPTUNA_RESULTS_FILE}")
    except Exception as e:
        print(f"无法保存试验结果数据框: {e}")


    # 可选：绘制优化历史和参数重要性图表
    if best_trial:  # 仅当存在最佳试验时才绘图
        try:
            # 检查可视化依赖项是否已安装
            import plotly
            import kaleido

            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_importance = optuna.visualization.plot_param_importances(study)

            history_path = os.path.join(OPTUNA_PLOTS_DIR, "optuna_ppo_history.png")
            importance_path = os.path.join(OPTUNA_PLOTS_DIR, "optuna_ppo_importance.png")

            fig_history.write_image(history_path)
            fig_importance.write_image(importance_path)
            
            # 可视化超参相关性
            fig_contour = optuna.visualization.plot_contour(study, params=["learning_rate", "gamma"])
            contour_path = os.path.join(OPTUNA_PLOTS_DIR, "optuna_ppo_contour.png")
            fig_contour.write_image(contour_path)
            
            print(f"Optuna图表已保存到 {OPTUNA_PLOTS_DIR}")
        except ImportError:
            print("\n安装plotly和kaleido以生成Optuna图表: pip install plotly kaleido")
        except Exception as e:
            print(f"\n无法生成Optuna图表: {e}")