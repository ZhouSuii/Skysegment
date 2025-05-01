import optuna
import json
import torch
import numpy as np
from tqdm import tqdm
import time
import os
import traceback # 导入 traceback

from new_environment import GraphPartitionEnvironment
from agent_gnn import GNNDQNAgent # 确保导入的是更新后的 Agent
from metrics import evaluate_partition, calculate_partition_weights # 导入 calculate_partition_weights
from run_experiments import create_test_graph

# --- 配置 ---
NUM_NODES = 20                  # 稍微增大图规模以获得更泛化的参数
NUM_PARTITIONS = 2
N_TRIALS = 150                  # 根据时间和资源调整
TRAINING_EPISODES = 600         # 增加训练回合数
MAX_STEPS_PER_EPISODE = 50
OPTUNA_TIMEOUT = 3600 * 4       # 增加超时时间 (例如, 4小时)
RESULTS_DIR = "results"
CONFIGS_DIR = "configs"
BEST_PARAMS_FILE = os.path.join(CONFIGS_DIR, "best_gnn_dqn_params.json") # 使用新文件名
OPTUNA_DB_NAME = "optuna_gnn_dqn_study.db" # 使用新数据库名
OPTUNA_RESULTS_FILE = os.path.join(RESULTS_DIR, "optuna_gnn_dqn_results.csv")
OPTUNA_PLOTS_DIR = os.path.join(RESULTS_DIR, "tune_gnn_plots")
REPORT_INTERVAL = 50            # 每隔多少回合报告一次中间目标值给 Optuna Pruner
IMBALANCE_PENALTY = 0.1         # 不平衡惩罚系数
DEFAULT_CONFIG_PATH = "configs/default.json" # 定义默认配置文件路径

# 确保目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(OPTUNA_PLOTS_DIR, exist_ok=True)

# --- Optuna目标函数 ---
def objective(trial: optuna.Trial):
    """GNN-DQN超参数调优的Optuna目标函数 (改进版)"""
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

    potential_weights_to_use = loaded_potential_weights if loaded_potential_weights else default_pbrs_weights
    if not loaded_potential_weights:
         print(f"试验 {trial.number}: 使用默认 PBRS 权重: {potential_weights_to_use}")

    # 1. 建议超参数 (扩展和调整搜索空间)
    config = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True), # 调整范围
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 384, 512]), # 增加选项
        'num_layers': trial.suggest_int('num_layers', 2, 5), # GNN层数 (增加范围)
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.6), # Dropout率 (增加上限)
        'gamma': trial.suggest_float('gamma', 0.95, 0.999, log=True), # 折扣因子 (调整范围和分布)
        'epsilon_decay': trial.suggest_float('epsilon_decay', 0.995, 0.9999), # 探索率衰减 (更慢的衰减)
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]), # 批量大小
        'target_update_freq': trial.suggest_int('target_update_freq', 10, 100), # 目标网络更新频率 (扩大范围)
        'memory_size': trial.suggest_categorical('memory_size', [5000, 10000, 20000]), # 经验回放容量
        # (可选) 添加 Adam 优化器参数
        'adam_beta1': trial.suggest_float('adam_beta1', 0.85, 0.95),
        'adam_beta2': trial.suggest_float('adam_beta2', 0.99, 0.9999),
        # 固定参数
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'jit_compile': False, # JIT 对 GNN 帮助可能有限
        'use_cuda_streams': True # 假设 Agent 内部会利用
    }
    print(f"试验 {trial.number}: 配置 = {config}")

    agent = None
    env = None
    graph = None # 提前声明
    try:
        # 2. 设置环境和智能体
        # 为减少方差，可以在所有试验中使用相同的几个图，或者像现在这样每次生成
        graph = create_test_graph(num_nodes=NUM_NODES, seed=trial.number % 10) # 使用模运算复用种子
        env = GraphPartitionEnvironment(
            graph,
            NUM_PARTITIONS,
            max_steps=MAX_STEPS_PER_EPISODE,
            gamma=config['gamma'],
            potential_weights=potential_weights_to_use # 使用获取到的权重
        )
        agent = GNNDQNAgent(graph, NUM_PARTITIONS, config=config) # 确保传入 config

        # 3. 训练循环
        best_intermediate_objective = float('inf') # 记录训练过程中的最佳目标值
        final_partition = None # 记录达到最佳目标值时的分区

        # --- GPU 预热 ---
        if device.type == 'cuda':
            print(f"试验 {trial.number}: 预热 GPU...")
            # 执行一些小批量的前向和后向传播来预热GPU
            temp_state, _ = env.reset()
            temp_agent = GNNDQNAgent(graph, NUM_PARTITIONS, config=config) # 创建临时 agent
            for _ in range(agent.batch_size * 2 // MAX_STEPS_PER_EPISODE + 1): # 确保收集足够样本
                temp_state_inner, _ = env.reset()
                for _ in range(MAX_STEPS_PER_EPISODE):
                    temp_action = temp_agent.act(temp_state_inner) # 使用 agent 的 act
                    temp_next_state, temp_reward, temp_done, _, _ = env.step(temp_action)
                    temp_agent.remember(temp_state_inner, temp_action, temp_reward, temp_next_state, temp_done)
                    temp_state_inner = temp_next_state
                    if temp_done: break
            if temp_agent.current_memory_size >= temp_agent.batch_size:
                temp_agent.replay() # 进行一次学习
            del temp_agent # 删除临时 agent
            torch.cuda.synchronize() # 等待预热完成
            print(f"试验 {trial.number}: GPU 预热完成。")
        # --- 预热结束 ---

        # 使用tqdm显示训练进度
        pbar = tqdm(range(TRAINING_EPISODES), desc=f"Trial {trial.number}")
        for e in pbar:
            state, _ = env.reset()
            total_reward = 0
            done = False
            episode_loss = 0.0
            steps_in_episode = 0

            for step in range(MAX_STEPS_PER_EPISODE):
                action = agent.act(state)
                next_state, reward, done, _, info = env.step(action)
                # 存储经验 (确保 reward 是 float)
                agent.remember(state, action, float(reward), next_state, done)
                state = next_state
                total_reward += reward
                steps_in_episode += 1

                # 经验回放 (Agent 内部处理批量大小检查)
                loss = agent.replay()
                if loss is not None:
                    episode_loss += loss

                if done:
                    break

            avg_loss = episode_loss / max(1, agent.train_count if hasattr(agent, 'train_count') and agent.train_count > 0 else steps_in_episode) # 使用 agent 的训练计数更准确
            if hasattr(agent, 'train_count'): agent.train_count = 0 # 重置计数器

            pbar.set_postfix({"Reward": f"{total_reward:.2f}", "Loss": f"{avg_loss:.4f}", "Epsilon": f"{agent.epsilon:.3f}"})

            # --- 改进的剪枝报告 ---
            # 定期计算并报告中间目标值
            if (e + 1) % REPORT_INTERVAL == 0 or e == TRAINING_EPISODES - 1:
                current_partition = env.partition_assignment # 获取当前回合结束时的分区
                # 检查分区是否有效 (所有分区都被分配了节点)
                if current_partition is not None and len(np.unique(current_partition)) == NUM_PARTITIONS:
                    eval_results_intermediate = evaluate_partition(graph, current_partition, NUM_PARTITIONS, print_results=False)
                    # 计算目标值
                    intermediate_objective = eval_results_intermediate["normalized_cut"] + IMBALANCE_PENALTY * max(0, eval_results_intermediate["weight_imbalance"] - 1) # 惩罚项不小于0

                    # 报告给 Optuna Pruner (我们希望最小化目标值)
                    trial.report(intermediate_objective, e)

                    # 更新本试验内记录的最佳分区
                    if intermediate_objective < best_intermediate_objective:
                        best_intermediate_objective = intermediate_objective
                        final_partition = current_partition.copy() # 保存最佳分区状态

                    # 检查是否应提前停止试验
                    if trial.should_prune():
                        print(f"试验 {trial.number}: 在回合 {e+1} 被剪枝 (中间目标值: {intermediate_objective:.4f})。")
                        # 对于被剪枝的试验，返回当前记录的最佳中间目标值
                        # 或者返回一个特定的高值，表明它被剪枝了
                        return best_intermediate_objective if best_intermediate_objective != float('inf') else float('inf') # 返回记录的最佳值
                else:
                    # 如果分区无效或未完成，报告一个高值
                    trial.report(float('inf'), e) # 报告一个高值表示当前状态不佳
                    if trial.should_prune():
                         print(f"试验 {trial.number}: 在回合 {e+1} 因无效/未完成分区被剪枝。")
                         return float('inf') # 返回高值

        # 4. 最终评估
        # 使用训练过程中记录的最佳分区进行最终评估
        if final_partition is None:
            # 如果从未记录过有效分区 (可能训练早期就失败或从未达到报告点)
            # 尝试使用最后的分区
            final_partition = env.partition_assignment
            if final_partition is None or len(np.unique(final_partition)) < NUM_PARTITIONS:
                print(f"试验 {trial.number}: 未能生成有效的最终分区。")
                return float('inf') # 返回高值表示失败

        # 确保分区有效性 (冗余检查)
        if len(np.unique(final_partition)) < NUM_PARTITIONS:
            print(f"试验 {trial.number}: 最终记录的分区无效。")
            return float('inf')

        eval_results = evaluate_partition(graph, final_partition, NUM_PARTITIONS, print_results=False)
        print(f"试验 {trial.number}: 最终评估结果 = {eval_results}")

        # 定义目标：最小化归一化割边 + 不平衡惩罚
        objective_value = eval_results["normalized_cut"] + IMBALANCE_PENALTY * max(0, eval_results["weight_imbalance"] - 1)
        print(f"试验 {trial.number}: 最终目标值 = {objective_value:.6f}")

        # 返回最终计算的目标值
        return objective_value

    except optuna.exceptions.TrialPruned:
        # 如果是因为剪枝而退出，返回记录的最佳中间目标值
        print(f"试验 {trial.number}: 确认剪枝退出。")
        return best_intermediate_objective if best_intermediate_objective != float('inf') else float('inf')
    except Exception as e:
        print(f"试验 {trial.number}: 失败，错误: {e}")
        traceback.print_exc() # 打印详细错误信息
        return float('inf') # 返回大值表示失败

    finally:
        # 5. 清理
        del agent
        del env
        del graph # 确保图对象也被删除
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"试验 {trial.number}: 完成。用时: {time.time() - run_start_time:.2f}s")


# --- 主执行 ---
if __name__ == "__main__":
    study_name = "gnn-dqn-partitioning-optimization-v3" # 使用新名称

    # 使用MedianPruner，基于更相关的目标值进行剪枝
    # n_warmup_steps 现在对应于报告的步数 (e)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, # 增加启动试验次数
        n_warmup_steps=TRAINING_EPISODES // REPORT_INTERVAL // 2, # 基于报告次数设置预热步数 (例如，预热到一半的报告点)
        interval_steps=1 # 每报告一次就检查剪枝
    )

    # --- 使用 SQLite 存储 ---
    storage_name = f"sqlite:///{OPTUNA_DB_NAME}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name, # 指定存储
        load_if_exists=True, # 如果数据库存在则加载
        direction='minimize',
        pruner=pruner
    )

    print(f"开始GNN-DQN的Optuna优化，共{N_TRIALS}次试验 (超时: {OPTUNA_TIMEOUT}s)...")
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
        traceback.print_exc() # 打印详细错误

    # --- 结果处理 ---
    print("\n" + "="*20 + " 优化完成! " + "="*20)
    print(f"数据库: {storage_name}")
    print(f"完成的试验数: {len(study.trials)}")

    # 查找并打印最佳试验
    try:
        best_trial = study.best_trial
        print("\n最佳试验:")
        print(f"  编号: {best_trial.number}")
        print(f"  值 (目标): {best_trial.value:.6f}") # 打印目标值
        print("  参数: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # 保存最佳超参数
        with open(BEST_PARAMS_FILE, 'w') as f:
            json.dump(best_trial.params, f, indent=4)
        print(f"\n最佳超参数已保存到 {BEST_PARAMS_FILE}")

    except ValueError:
        print("\n警告: 没有试验成功完成或数据库为空。无法确定最佳试验。")
        # 尝试从数据库中加载最佳试验（如果存在）
        try:
            loaded_study = optuna.load_study(study_name=study_name, storage=storage_name)
            if loaded_study.best_trial:
                best_trial = loaded_study.best_trial
                print("\n从数据库加载的最佳试验:")
                # ... (打印信息的代码同上) ...
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
        # 确保存储路径正确
        optuna_results_path = os.path.join(RESULTS_DIR, f"{study_name}_results.csv")
        results_df.to_csv(optuna_results_path, index=False)
        print(f"Optuna试验结果已保存到 {optuna_results_path}")
    except Exception as e:
        print(f"无法保存试验结果数据框: {e}")


    # 可选：绘制优化历史和参数重要性图表
    if best_trial: # 只有在确定有最佳试验时才绘图
        try:
            # 检查可视化依赖项
            import plotly
            import kaleido # kaleido 用于静态图片导出

            print("正在生成Optuna可视化图表...")
            # 确保存储路径正确
            history_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_history.png")
            importance_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_importance.png")
            parallel_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_parallel.png")
            slice_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_slice.png")
            contour_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_contour.png") # 添加等高线图

            # 生成图表对象
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
            fig_slice = optuna.visualization.plot_slice(study) # 切片图
            # 选择两个最重要的参数绘制等高线图
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


            # 保存图表为静态图片
            fig_history.write_image(history_path)
            fig_importance.write_image(importance_path)
            fig_parallel.write_image(parallel_path)
            fig_slice.write_image(slice_path)

            print(f"Optuna图表已保存到 {OPTUNA_PLOTS_DIR}")

        except ImportError:
            print("\n警告: 请安装plotly和kaleido以生成Optuna图表: pip install plotly kaleido")
        except Exception as e:
            print(f"\n无法生成Optuna图表: {e}")
            traceback.print_exc() # 打印详细错误