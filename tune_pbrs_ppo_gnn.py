import optuna
import json
import torch
# import networkx as nx # 不再直接使用
import numpy as np
from tqdm import tqdm
import time
import os
import traceback # 导入 traceback
import pandas as pd
import math # 用于检测 inf 和 nan 值

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
OPTUNA_PLOTS_DIR = os.path.join(RESULTS_DIR, "optuna_pbrs_ppo_gnn_plots")
REPORT_INTERVAL = 50 # 每隔多少回合报告一次中间目标值
# 以下权重将改为由Optuna调优，不再使用全局固定值
# IMBALANCE_PENALTY = 0.5 # 不平衡惩罚系数
# MODULARITY_WEIGHT = 1.0  # 调整这个权重以反映模块度的重要性

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
    'gamma': trial.suggest_float('gamma', 0.97, 0.999, log=True),  # 最重要
    'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
    'clip_ratio': trial.suggest_float('clip_ratio', 0.1, 0.3),
    'value_coef': trial.suggest_float('value_coef', 0.4, 0.6),
    'entropy_coef': trial.suggest_float('entropy_coef', 1e-4, 0.05, log=True),
    'adam_beta1': trial.suggest_float('adam_beta1', 0.85, 0.95),
    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
    # 固定不太重要的参数
    'hidden_dim': 256,        # 使用最佳值
    'num_layers': 3,          # 使用最佳值
    'ppo_epochs': 10,         # 使用最佳值
    'batch_size': 256,        
    'adam_beta2': 0.99,       # 使用最佳值
    'update_frequency': 2048
}

    # 建议 PBRS 势能函数权重
    potential_weights = {
    'variance': trial.suggest_float('pbrs_variance_weight', 0.1, 4, log=True),  # 降低上限
    'edge_cut': trial.suggest_float('pbrs_edge_cut_weight', 0.5, 5.0, log=True),  # 提高下限
    'modularity': trial.suggest_float('pbrs_modularity_weight', 0.5, 5.0, log=True)  # 提高下限
}

    # 建议目标函数权重作为超参数
    obj_ncut_weight = trial.suggest_float('obj_ncut_weight', 0.1, 2.0, log=True)  # 归一化切边权重
    obj_modularity_weight = trial.suggest_float('obj_modularity_weight', 0.1, 2.0, log=True)  # 模块度权重
    obj_imbalance_penalty = trial.suggest_float('obj_imbalance_penalty', 0.1, 2.0, log=True)  # 不平衡惩罚系数
    
    print(f"试验 {trial.number}: 配置 = {config}, PBRS 权重 = {potential_weights}")
    print(f"试验 {trial.number}: 目标函数权重: 归一化切边={obj_ncut_weight:.3f}, 模块度={obj_modularity_weight:.3f}, 不平衡惩罚={obj_imbalance_penalty:.3f}")

    agent = None
    env = None
    graph = None
    total_edges = 0  # 初始化 total_edges 变量
    try:
        # 2. 设置环境和智能体
        graph = create_test_graph(num_nodes=NUM_NODES, seed=trial.number % 10) # 复用种子
        total_edges = len(graph.edges())  # 设置图边的数量
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
                    # 使用动态权重计算中间目标值
                    intermediate_objective = (obj_ncut_weight * eval_results_intermediate["normalized_cut"] - 
                                             obj_modularity_weight * eval_results_intermediate["modularity"] + 
                                             obj_imbalance_penalty * max(0, eval_results_intermediate["weight_imbalance"] - 1))

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

        # 计算最终目标值 - 使用Optuna建议的权重而非全局常量
        objective_value = (obj_ncut_weight * eval_results["normalized_cut"] -
                           obj_modularity_weight * eval_results["modularity"] +
                           obj_imbalance_penalty * max(0, eval_results["weight_imbalance"] - 1))

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


# 添加过滤无效试验的可视化函数
def create_visualization_with_valid_trials(study, save_path=None):
    """
    创建只包含有效试验的可视化 (排除目标值为 inf 或 nan 的试验)
    
    Args:
        study: Optuna study 对象
        save_path: 图片保存路径 (可选)
    
    Returns:
        包含有效试验的 Plotly 图形对象
    """
    try:
        # 获取所有试验数据
        trials_df = study.trials_dataframe()
        if trials_df.empty:
            print("警告: 没有可用的试验数据")
            return None
        
        # 过滤掉目标值为 inf 或 nan 的试验
        valid_trials_mask = ~(trials_df["value"].isnull() | 
                             trials_df["value"].apply(lambda x: math.isinf(x) if isinstance(x, float) else False))
        valid_trials_df = trials_df[valid_trials_mask]
        
        # 检查是否有有效试验
        if valid_trials_df.empty:
            print("警告: 所有试验的目标值都是 inf 或 nan")
            return None
            
        print(f"总试验数: {len(trials_df)}, 有效试验数: {len(valid_trials_df)}, 过滤掉的试验数: {len(trials_df) - len(valid_trials_df)}")
        
        # 获取最佳试验
        best_trial_in_valid = valid_trials_df.loc[valid_trials_df["value"].idxmin()]
        
        # 获取参数名称
        param_names = [col for col in valid_trials_df.columns if col.startswith("params_")]
        param_labels = [col.replace("params_", "") for col in param_names]
        
        # 创建平行坐标图
        fig = go.Figure()
        
        # 创建维度列表
        dimensions = []
        for param in param_names:
            param_values = valid_trials_df[param]
            dimensions.append(
                dict(
                    range=[param_values.min(), param_values.max()],
                    label=param.replace("params_", ""),
                    values=param_values
                )
            )
        
        # 添加目标值维度
        dimensions.append(
            dict(
                range=[valid_trials_df["value"].min(), valid_trials_df["value"].max()],
                label="Objective Value",
                values=valid_trials_df["value"]
            )
        )
        
        # 添加平行坐标图
        fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=valid_trials_df["value"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Objective Value"),
                ),
                dimensions=dimensions,
                labelangle=30,
                labelside='bottom'
            )
        )
        
        # 更新布局
        fig.update_layout(
            title="Parameter visualization of valid trials (excluding inf/nan target values)",
            font=dict(size=12),
            height=600,
            width=1000,
            margin=dict(l=80, r=80, t=80, b=80),
        )
        
        # 保存图表（如果指定了保存路径）
        if save_path:
            # 判断文件扩展名
            is_html = save_path.lower().endswith('.html')
            
            if is_html:
                try:
                    fig.write_html(save_path)
                    print(f"有效试验可视化已保存为HTML格式: {save_path}")
                except Exception as html_error:
                    print(f"保存HTML失败: {html_error}")
                    raise html_error
            else:
                try:
                    fig.write_image(save_path)
                    print(f"有效试验可视化已保存为图像格式: {save_path}")
                except Exception as render_error:
                    print(f"保存图像失败 (可能是 Plotly/Kaleido 渲染问题): {render_error}")
                    # 尝试将图表保存为HTML格式，这通常更可靠
                    try:
                        html_path = os.path.splitext(save_path)[0] + '.html'
                        fig.write_html(html_path)
                        print(f"已改为将图表保存为HTML格式: {html_path}")
                    except Exception as html_error:
                        print(f"保存为HTML也失败: {html_error}")
                        raise render_error  # 重新抛出原始错误
        
        return fig
        
    except Exception as e:
        print(f"创建有效试验可视化时出错: {e}")
        traceback.print_exc()
        return None


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

    # 替换原有的平行坐标图生成代码

    # 可选：绘制图表 (与 DQN 脚本类似)
    if best_trial:
        try:
            import plotly
            import kaleido
            import plotly.graph_objects as go
            import plotly.io as pio
            
            # 尝试设置渲染引擎为 'svg'，这可能在某些环境下更可靠
            try:
                pio.kaleido.scope.default_format = "svg"
                print("已将Plotly渲染引擎设置为SVG格式")
            except Exception as e:
                print(f"设置Plotly渲染引擎时出错: {e}")
                
            print("正在生成Optuna可视化图表...")
            # 使用SVG格式替代PNG
            history_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_history.svg")
            importance_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_importance.svg") 
            parallel_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_parallel.svg")
            slice_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_slice.svg")
            contour_path = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_contour.svg")
            
            # 同时保存HTML版本以便交互
            history_html = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_history.html")
            importance_html = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_importance.html")
            parallel_html = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_parallel.html")
            slice_html = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_slice.html")
            contour_html = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_contour.html")

            # 生成标准图表
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_importance = optuna.visualization.plot_param_importances(study)
            fig_slice = optuna.visualization.plot_slice(study)
            
            # 获取参数重要性
            try:
                param_importances = optuna.importance.get_param_importances(study)
                print("参数重要性排序：")
                for param, importance in param_importances.items():
                    print(f"  {param}: {importance:.4f}")
                    
                # 创建改进的平行坐标图
                # 1. 按重要性排序参数
                sorted_params = list(param_importances.keys())
                
                # 2. 如果参数太多，只选择最重要的N个
                MAX_PARAMS = 8  # 最多显示的参数数量
                if len(sorted_params) > MAX_PARAMS:
                    print(f"平行坐标图中仅显示最重要的 {MAX_PARAMS} 个参数")
                    sorted_params = sorted_params[:MAX_PARAMS]
                
                # 3. 创建自定义平行坐标图
                trials_df = study.trials_dataframe()
                if not trials_df.empty:
                    # 将sorted_params作为维度，按重要性排序
                    dimensions = [dict(
                        range=[trials_df[f"params_{param}"].min(), trials_df[f"params_{param}"].max()],
                        label=param,
                        values=trials_df[f"params_{param}"]
                    ) for param in sorted_params if f"params_{param}" in trials_df.columns]
                    
                    # 创建一个新的平行坐标图
                    fig_parallel_custom = go.Figure()
                    
                    # 添加非最佳试验 - 移除不支持的opacity属性
                    fig_parallel_custom.add_trace(
                        go.Parcoords(
                            line=dict(
                                color=trials_df["value"],
                                colorscale="Viridis",
                                showscale=True,
                                colorbar=dict(title="Objective Value"),
                                # opacity在Parcoords中不支持，移除此设置
                            ),
                            dimensions=dimensions,
                            labelangle=30,  # 倾斜标签以便更好地阅读
                            labelside='bottom',  # 将标签放在顶部
                        )
                    )
                    
                    # 突出显示最佳试验
                    best_trial_values = [trials_df.loc[trials_df["number"] == best_trial.number, f"params_{param}"].values[0] 
                                        for param in sorted_params if f"params_{param}" in trials_df.columns]
                    
                    fig_parallel_custom.add_trace(
                        go.Scattergl(
                            x=list(range(len(dimensions))),
                            y=best_trial_values,
                            mode="lines+markers",
                            line=dict(color="red", width=4),
                            marker=dict(size=10, color="red", symbol="star"),
                            name="最佳试验"
                        )
                    )
                    
                    # 更新布局
                    fig_parallel_custom.update_layout(
                        title="Parallel Coordinate Plot of Parameter Importance",
                        font=dict(size=12),
                        height=600,
                        width=1000,
                        margin=dict(l=80, r=80, t=80, b=80),
                    )
                    
                    # 保存自定义平行坐标图，添加错误处理
                    # 总是保存HTML版本（交互式）
                    try:
                        fig_parallel_custom.write_html(parallel_html)
                        print(f"自定义平行坐标图已保存为交互式HTML格式: {parallel_html}")
                    except Exception as html_error:
                        print(f"保存交互式HTML版本失败: {html_error}")
                    
                    # 尝试保存SVG静态图像
                    try:
                        fig_parallel_custom.write_image(parallel_path)
                        print(f"自定义平行坐标图已保存为SVG格式: {parallel_path}")
                    except Exception as svg_error:
                        print(f"保存SVG图像失败: {svg_error}")
                
                # 保存标准图表，添加错误处理
                # 同时保存HTML和SVG格式
                for fig_item, svg_path, html_path, name in [
                    (fig_history, history_path, history_html, "优化历史"),
                    (fig_importance, importance_path, importance_html, "参数重要性"),
                    (fig_slice, slice_path, slice_html, "参数切片")
                ]:
                    # 总是先尝试保存HTML（交互式）版本
                    try:
                        fig_item.write_html(html_path)
                        print(f"{name}图表已保存为交互式HTML格式: {html_path}")
                    except Exception as html_error:
                        print(f"保存{name}图表HTML版本失败: {html_error}")
                    
                    # 再尝试保存SVG静态图像
                    try:
                        fig_item.write_image(svg_path)
                        print(f"{name}图表已保存为SVG格式: {svg_path}")
                    except Exception as svg_error:
                        print(f"保存{name}图表SVG版本失败: {svg_error}")
                
                # 生成等高线图（仅使用最重要的两个参数）
                if len(param_importances) >= 2:
                    top_two_params = list(param_importances.keys())[:2]
                    try:
                        fig_contour = optuna.visualization.plot_contour(study, params=top_two_params)
                        
                        # 总是先保存HTML（交互式）版本
                        try:
                            fig_contour.write_html(contour_html)
                            print(f"等高线图表已保存为交互式HTML格式: {contour_html}")
                        except Exception as html_error:
                            print(f"保存等高线图表HTML版本失败: {html_error}")
                        
                        # 再尝试保存SVG静态图像
                        try:
                            fig_contour.write_image(contour_path)
                            print(f"等高线图表已保存为SVG格式: {contour_path}")
                        except Exception as svg_error:
                            print(f"保存等高线图表SVG版本失败: {svg_error}")
                    except Exception as e:
                        print(f"生成等高线图表失败: {e}")
                else:
                    print("参数不足2个，无法生成等高线图。")
                    
                # 创建有效试验的可视化
                # 使用SVG和HTML格式
                valid_trials_svg = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_valid_trials.svg")
                valid_trials_html = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_valid_trials.html")
                
                # 首先尝试直接保存HTML版本
                try:
                    fig_valid = create_visualization_with_valid_trials(study, save_path=None)
                    if fig_valid:
                        fig_valid.write_html(valid_trials_html)
                        print(f"有效试验可视化已保存为交互式HTML格式: {valid_trials_html}")
                        
                        # 再尝试保存SVG版本
                        try:
                            fig_valid.write_image(valid_trials_svg)
                            print(f"有效试验可视化已保存为SVG格式: {valid_trials_svg}")
                        except Exception as svg_error:
                            print(f"保存有效试验可视化为SVG格式失败: {svg_error}")
                except Exception as viz_error:
                    print(f"创建有效试验可视化时出错: {viz_error}")
                    # 保存结果到CSV作为备选方案
                    valid_trials_csv = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_valid_trials.csv")
                    try:
                        trials_df = study.trials_dataframe()
                        # 过滤无效值
                        valid_trials_mask = ~(trials_df["value"].isnull() | 
                                           trials_df["value"].apply(lambda x: math.isinf(x) if isinstance(x, float) else False))
                        valid_trials_df = trials_df[valid_trials_mask]
                        valid_trials_df.to_csv(valid_trials_csv, index=False)
                        print(f"有效试验数据已保存到CSV: {valid_trials_csv}")
                    except Exception as csv_error:
                        print(f"保存有效试验到CSV也失败: {csv_error}")
                
                print(f"Optuna图表已保存到 {OPTUNA_PLOTS_DIR}")
                
            except Exception as importance_err:
                print(f"获取参数重要性失败: {importance_err}")
                # 退回到标准平行坐标图
                try:
                    fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
                    fig_parallel.write_image(parallel_path)
                except Exception as parallel_err:
                    print(f"生成标准平行坐标图失败: {parallel_err}")
                
                # 即使参数重要性失败，仍尝试创建有效试验的可视化
                valid_trials_svg = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_valid_trials.svg")
                valid_trials_html = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_valid_trials.html")
                
                try:
                    fig_valid = create_visualization_with_valid_trials(study, save_path=None)
                    if fig_valid:
                        # 首先尝试保存HTML版本
                        try:
                            fig_valid.write_html(valid_trials_html)
                            print(f"有效试验可视化已保存为交互式HTML格式: {valid_trials_html}")
                        except Exception as html_error:
                            print(f"保存有效试验HTML失败: {html_error}")
                            
                        # 再尝试保存SVG版本
                        try:
                            fig_valid.write_image(valid_trials_svg)
                            print(f"有效试验可视化已保存为SVG格式: {valid_trials_svg}")
                        except Exception as svg_error:
                            print(f"保存有效试验SVG失败: {svg_error}")
                except Exception as viz_err:
                    print(f"创建有效试验可视化也失败: {viz_err}")
                    # 保存CSV作为备选方案
                    valid_trials_csv = os.path.join(OPTUNA_PLOTS_DIR, f"{study_name}_valid_trials.csv")
                    try:
                        trials_df = study.trials_dataframe()
                        valid_trials_mask = ~(trials_df["value"].isnull() | 
                                           trials_df["value"].apply(lambda x: math.isinf(x) if isinstance(x, float) else False))
                        valid_trials_df = trials_df[valid_trials_mask]
                        valid_trials_df.to_csv(valid_trials_csv, index=False)
                        print(f"有效试验数据已保存到CSV: {valid_trials_csv}")
                    except Exception as csv_error:
                        print(f"保存有效试验到CSV也失败: {csv_error}")
                
        except ImportError:
            print("\n警告: 请安装plotly和kaleido以生成Optuna图表: pip install plotly kaleido")
        except Exception as e:
            print(f"\n无法生成Optuna图表: {e}")
            traceback.print_exc()