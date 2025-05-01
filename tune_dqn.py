import optuna
import json
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
import time
import os

from new_environment import GraphPartitionEnvironment
from agent_dqn_basic import DQNAgent
from metrics import evaluate_partition
from run_experiments import create_test_graph

# --- 配置 ---
NUM_NODES = 10                  # 使用中等大小的图进行调优
NUM_PARTITIONS = 2
N_TRIALS = 200                  # Optuna试验次数
TRAINING_EPISODES = 300         # 每次试验的训练回合数(减少以加快调优)
MAX_STEPS_PER_EPISODE = 50      # 每回合最大步数
OPTUNA_TIMEOUT = 3600           # 整个优化过程的超时时间, 单位是秒s
RESULTS_DIR = "results"
CONFIGS_DIR = "configs"
BEST_PARAMS_FILE = os.path.join(CONFIGS_DIR, "best_dqn_params.json")
OPTUNA_RESULTS_FILE = os.path.join(RESULTS_DIR, "optuna_dqn_results.csv")
OPTUNA_PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
OPTUNA_DB_NAME = "optuna_dqn_study.db"
DEFAULT_CONFIG_PATH = "configs/default.json" # 定义默认配置文件路径

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(OPTUNA_PLOTS_DIR, exist_ok=True)

# --- Optuna目标函数 ---
def objective(trial: optuna.Trial):
    """DQN超参数调优的Optuna目标函数"""
    run_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"试验 {trial.number}: 开始于设备 {device}")

    # --- 修改：加载 default.json 获取 PBRS 权重 ---
    loaded_potential_weights = None
    default_pbrs_weights = {'variance': 1.0, 'edge_cut': 1.0, 'modularity': 1.0}
    try:
        if os.path.exists(DEFAULT_CONFIG_PATH):
            with open(DEFAULT_CONFIG_PATH, 'r') as f:
                default_config_data = json.load(f)
                if "potential_weights" in default_config_data:
                    loaded_potential_weights = default_config_data["potential_weights"]
                    print(f"试验 {trial.number}: 从 {DEFAULT_CONFIG_PATH} 加载 PBRS 权重: {loaded_potential_weights}")
                else:
                    print(f"警告: 在 {DEFAULT_CONFIG_PATH} 中未找到 'potential_weights' 键。")
        else:
            print(f"警告: 默认配置文件 {DEFAULT_CONFIG_PATH} 不存在。")
    except Exception as e:
        print(f"警告: 加载 {DEFAULT_CONFIG_PATH} 出错: {e}。")

    # 如果加载失败或未找到，则使用硬编码的默认值
    potential_weights_to_use = loaded_potential_weights if loaded_potential_weights else default_pbrs_weights
    if not loaded_potential_weights:
         print(f"试验 {trial.number}: 使用默认 PBRS 权重: {potential_weights_to_use}")

    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'hidden_sizes': trial.suggest_categorical('hidden_sizes', [[128, 128], [256, 128], [256, 256], [512, 256]]),
        'gamma': trial.suggest_float('gamma', 0.9, 0.99),  # 折扣因子
        'epsilon_decay': trial.suggest_float('epsilon_decay', 0.99, 0.999),  # 探索率衰减
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),  # 影响GPU利用率和内存
        'target_update_freq': trial.suggest_int('target_update_freq', 5, 20),  # 目标网络更新频率
        'memory_capacity': trial.suggest_categorical('memory_capacity', [2000, 5000, 10000]),  # 经验回放容量
        # 固定参数以提高调优效率
        'epsilon': 1.0,  # 初始探索率固定为1.0
        'epsilon_min': 0.01,  # 最小探索率固定
        'jit_compile': True,  # 启用JIT编译加速GPU计算
        'use_tensorboard': False  # 在调优过程中禁用TensorBoard，减少CPU开销
    }
    print(f"试验 {trial.number}: 配置 = {config}")

    agent = None
    env = None
    try:
        # 2. 设置环境和智能体
        # 为每次试验创建不同的图，避免过拟合到特定图结构
        graph = create_test_graph(num_nodes=NUM_NODES, seed=trial.number)
        env = GraphPartitionEnvironment(
            graph,
            NUM_PARTITIONS,
            max_steps=MAX_STEPS_PER_EPISODE,
            gamma=config['gamma'],
            potential_weights=potential_weights_to_use # 使用获取到的权重
        )

        # 计算状态和动作空间大小
        num_nodes = len(graph.nodes())
        state_size = num_nodes * (NUM_PARTITIONS + 2)  # 扁平化状态大小
        action_size = num_nodes * NUM_PARTITIONS

        agent = DQNAgent(state_size, action_size, config=config)

        # 3. 训练循环
        best_reward_trial = float('-inf')
        final_partition = None

        # 为减少CPU瓶颈，预先分配NumPy数组用于收集指标
        rewards = np.zeros(TRAINING_EPISODES)
        losses = np.zeros(TRAINING_EPISODES)

        # 使用tqdm显示训练进度
        for e in range(TRAINING_EPISODES):
            episode_start_time = time.time()
            state, _ = env.reset()
            total_reward = 0
            done = False
            episode_loss = 0.0
            steps_in_episode = 0

            for step in range(MAX_STEPS_PER_EPISODE):
                # 通过agent选择动作
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                
                # 存储经验
                agent.remember(state, action, float(reward), next_state, done)
                state = next_state
                total_reward += reward
                steps_in_episode += 1

                # 批量经验回放学习，减少CPU/GPU传输
                if agent.memory_counter >= agent.batch_size:
                    loss = agent.replay()
                    episode_loss += loss

                if done:
                    break

            # 计算平均损失
            avg_loss = episode_loss / max(1, steps_in_episode)
            rewards[e] = total_reward
            losses[e] = avg_loss

            # 更新最佳奖励
            if total_reward > best_reward_trial:
                best_reward_trial = total_reward
                final_partition = env.partition_assignment.copy()

            # Optuna剪枝：检查是否应提前停止试验
            trial.report(total_reward, e)
            if trial.should_prune():
                print(f"试验 {trial.number}: 在回合 {e} 被剪枝，奖励值 {total_reward}")
                raise optuna.exceptions.TrialPruned()

            if (e + 1) % 25 == 0:  # 定期打印进度
                print(f"试验 {trial.number} - 回合 {e+1}/{TRAINING_EPISODES} - 奖励: {total_reward:.2f}, 平均损失: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}, 用时: {time.time()-episode_start_time:.2f}s")

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
    # 创建或加载研究
    study_name = "dqn-partitioning-optimization"

    # 使用MedianPruner提前停止没有希望的试验
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=TRAINING_EPISODES // 4, interval_steps=10)

    # --- 修改：配置 SQLite 存储 ---
    storage_name = f"sqlite:///{OPTUNA_DB_NAME}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name, # 指定存储
        load_if_exists=True, # 如果数据库存在则加载
        direction='minimize',
        pruner=pruner
    )
    # --- 修改结束 ---

    print(f"开始DQN的Optuna优化，共{N_TRIALS}次试验 (超时: {OPTUNA_TIMEOUT}s)...")
    print(f"数据库存储: {storage_name}")
    print(f"图大小: {NUM_NODES}个节点, {NUM_PARTITIONS}个分区")
    print(f"每次试验训练: {TRAINING_EPISODES}个回合, {MAX_STEPS_PER_EPISODE}步/回合")

    try:
        # 设置gc_after_trial=True启用垃圾回收，减少内存占用
        study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, gc_after_trial=True)
    except KeyboardInterrupt:
        print("优化被手动停止。")
    except Exception as e:
        print(f"优化过程中发生错误: {e}")

    # --- 结果 ---
    print("\n" + "="*20 + " 优化完成! " + "="*20)
    print(f"数据库: {storage_name}")
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

            history_path = os.path.join(OPTUNA_PLOTS_DIR, "optuna_dqn_history.png")
            importance_path = os.path.join(OPTUNA_PLOTS_DIR, "optuna_dqn_importance.png")

            fig_history.write_image(history_path)
            fig_importance.write_image(importance_path)
            print(f"Optuna图表已保存到 {OPTUNA_PLOTS_DIR}")

            # 并行坐标图可视化超参数关系
            fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
            parallel_path = os.path.join(OPTUNA_PLOTS_DIR, "optuna_dqn_parallel.png")
            fig_parallel.write_image(parallel_path)

        except ImportError:
            print("\n安装plotly和kaleido以生成Optuna图表: pip install plotly kaleido")
        except Exception as e:
            print(f"\n无法生成Optuna图表: {e}")