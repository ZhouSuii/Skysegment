# view_partitions.py -- 对比各个算法的划分结果
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from new_environment import GraphPartitionEnvironment  # 使用更新的环境
from baselines import random_partition, weighted_greedy_partition, spectral_partition, metis_partition
from run_experiments import create_test_graph

# === 新增：导入所有需要的Agent ===
from agent_dqn_basic import DQNAgent
from agent_gnn import GNNDQNAgent
try:
    from agent_ppo import PPOAgent
    from agent_ppo_gnn_simple import SimplePPOAgentGNN as GNNPPOAgent
    ppo_available = True
except ImportError as e:
    print(f"⚠️ 导入PPO相关模块时出错，将跳过PPO模型加载。错误详情: {e}")
    ppo_available = False

def load_real_airspace_graph(graphml_path):
    """
    加载真实空域图
    
    Args:
        graphml_path: GraphML文件路径
    
    Returns:
        NetworkX图对象
    """
    try:
        print(f"🔄 加载真实空域图: {graphml_path}")
        G = nx.read_graphml(graphml_path)
        
        # === 修复：重新编号节点确保连续性 ===
        print(f"原始节点: {list(G.nodes())[:5]}...")  # 显示前5个节点
        
        # 创建节点映射
        node_mapping = {old_node: i for i, old_node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, node_mapping)
        
        print(f"重新编号后节点: {list(G.nodes())[:5]}...")
        
        # 确保所有节点属性都是正确的数值类型
        for node in G.nodes():
            # 权重属性转换
            if 'weight' in G.nodes[node]:
                G.nodes[node]['weight'] = float(G.nodes[node]['weight'])
            else:
                G.nodes[node]['weight'] = 1.0  # 默认权重
            
            # 坐标属性转换（如果存在）
            if 'lon' in G.nodes[node]:
                G.nodes[node]['lon'] = float(G.nodes[node]['lon'])
            if 'lat' in G.nodes[node]:
                G.nodes[node]['lat'] = float(G.nodes[node]['lat'])
        
        # === 修复：检查和修复图连通性 ===
        if G.number_of_edges() == 0:
            print("⚠️  图没有边，正在添加边...")
            G = add_edges_to_graph(G)
        
        if not nx.is_connected(G):
            print("⚠️  图不连通，正在修复连通性...")
            G = ensure_graph_connectivity(G)
        
        print(f"✅ 成功加载并修复真实空域图:")
        print(f"   节点数: {G.number_of_nodes()}")
        print(f"   边数: {G.number_of_edges()}")
        print(f"   连通性: {'是' if nx.is_connected(G) else '否'}")
        
        # 检查权重分布
        weights = [G.nodes[node]['weight'] for node in G.nodes()]
        print(f"   权重范围: [{min(weights):.2f}, {max(weights):.2f}]")
        
        return G
        
    except FileNotFoundError:
        print(f"❌ 文件未找到: {graphml_path}")
        print("   将使用随机测试图作为替代")
        return None
    except Exception as e:
        print(f"❌ 加载图文件时出错: {e}")
        print("   将使用随机测试图作为替代")
        return None

def add_edges_to_graph(G):
    """为没有边的图添加边"""
    nodes = list(G.nodes())
    node_coords = []
    
    # 提取坐标
    for node in nodes:
        if 'lon' in G.nodes[node] and 'lat' in G.nodes[node]:
            node_coords.append([G.nodes[node]['lon'], G.nodes[node]['lat']])
        else:
            # 如果没有坐标，随机生成
            node_coords.append([np.random.random(), np.random.random()])
    
    node_coords = np.array(node_coords)
    
    # 尝试Delaunay三角剖分
    try:
        from scipy.spatial import Delaunay
        if len(nodes) >= 3:
            delaunay = Delaunay(node_coords)
            edges_added = 0
            
            for simplex in delaunay.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        u, v = nodes[simplex[i]], nodes[simplex[j]]
                        if not G.has_edge(u, v):
                            G.add_edge(u, v)
                            edges_added += 1
            
            print(f"   通过Delaunay添加了 {edges_added} 条边")
    except Exception as e:
        print(f"   Delaunay失败: {e}，使用KNN连接")
        # 回退到KNN连接
        from sklearn.neighbors import NearestNeighbors
        k = min(4, len(nodes) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(node_coords)
        distances, indices = nbrs.kneighbors(node_coords)
        
        edges_added = 0
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors[1:]:  # 跳过自己
                u, v = nodes[i], nodes[neighbor]
                if not G.has_edge(u, v):
                    G.add_edge(u, v)
                    edges_added += 1
        
        print(f"   通过KNN添加了 {edges_added} 条边")
    
    return G

def ensure_graph_connectivity(G):
    """确保图的连通性"""
    if nx.is_connected(G):
        return G
    
    # 获取所有连通分量
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G
    
    print(f"   发现 {len(components)} 个连通分量，正在连接...")
    
    # 将所有分量连接到最大的分量
    main_component = max(components, key=len)
    
    for component in components:
        if component == main_component:
            continue
        
        # 在两个分量之间添加一条边
        node1 = list(main_component)[0]
        node2 = list(component)[0]
        G.add_edge(node1, node2)
        main_component.update(component)
    
    print(f"   图现在已连通")
    return G

def get_graph_for_visualization(use_real_graph=True, real_graph_path=None):
    """
    获取用于可视化的图
    
    Args:
        use_real_graph: 是否使用真实图
        real_graph_path: 真实图文件路径
    
    Returns:
        tuple: (graph, graph_name, is_real_graph)
    """
    if use_real_graph and real_graph_path:
        # 尝试加载真实图
        real_graph = load_real_airspace_graph(real_graph_path)
        if real_graph is not None:
            return real_graph, f"真实空域图({real_graph.number_of_nodes()}节点)", True
    
    # 回退到测试图
    print("🔄 使用随机测试图")
    test_graph = create_test_graph(num_nodes=10, seed=42)
    return test_graph, "测试图(10节点)", False

def create_geographic_layout(graph):
    """
    如果图有地理坐标，创建基于地理位置的布局
    
    Args:
        graph: NetworkX图
    
    Returns:
        dict: 节点位置字典
    """
    if all('lon' in graph.nodes[node] and 'lat' in graph.nodes[node] for node in graph.nodes()):
        # 使用真实地理坐标
        pos = {}
        for node in graph.nodes():
            pos[node] = (graph.nodes[node]['lon'], graph.nodes[node]['lat'])
        print("📍 使用地理坐标布局")
        return pos
    else:
        # 使用spring布局
        print("🌸 使用Spring布局")
        return nx.spring_layout(graph, seed=42)

def calculate_layout_parameters(graph, is_real_graph):
    """
    根据图的特点计算布局参数
    
    Args:
        graph: NetworkX图
        is_real_graph: 是否为真实图
    
    Returns:
        dict: 布局参数
    """
    num_nodes = graph.number_of_nodes()
    
    if is_real_graph:
        # 真实图的参数
        node_size = max(200, min(800, 3000 // num_nodes))  # 自适应节点大小
        font_size = max(6, min(12, 80 // num_nodes))       # 自适应字体大小
        figsize = (18, 12) if num_nodes > 20 else (15, 10)
    else:
        # 测试图的参数
        node_size = 500
        font_size = 10
        figsize = (15, 10)
    
    return {
        'node_size': node_size,
        'font_size': font_size,
        'figsize': figsize
    }

# === 主程序 ===
def main():
    # === 配置选项 ===
    USE_REAL_GRAPH = True  # 设置为True使用真实图，False使用测试图
    REAL_GRAPH_PATH = 'ctu_airspace_graph_1900_2000_kmeans.graphml'  # 真实图文件路径
    
    # 获取图
    graph, graph_name, is_real_graph = get_graph_for_visualization(
        use_real_graph=USE_REAL_GRAPH,
        real_graph_path=REAL_GRAPH_PATH
    )
    
    # 设置分区数（可根据图的大小调整）
    #num_partitions = 3 if graph.number_of_nodes() > 15 else 2
    num_partitions = 2
    print(f"📊 使用 {num_partitions} 个分区进行划分")
    
    # 获取布局参数
    layout_params = calculate_layout_parameters(graph, is_real_graph)
    
    # 获取不同算法的划分结果
    print("\n🧮 计算各算法的划分结果...")
    partitions = {}
    
    # --- 修正：基线算法现在使用 baseline_graph ---
    baseline_graph = graph
    baseline_num_partitions = num_partitions

    try:
        partitions["Random"] = random_partition(baseline_graph, baseline_num_partitions)
        print("  ✅ Random完成")
    except Exception as e:
        print(f"  ❌ Random失败: {e}")
    
    try:
        partitions["Greedy"] = weighted_greedy_partition(baseline_graph, baseline_num_partitions)
        print("  ✅ Greedy完成")
    except Exception as e:
        print(f"  ❌ Greedy失败: {e}")
    
    try:
        partitions["Spectral"] = spectral_partition(baseline_graph, baseline_num_partitions)
        print("  ✅ Spectral完成")
    except Exception as e:
        print(f"  ❌ Spectral失败: {e}")
    
    try:
        partitions["METIS"] = metis_partition(baseline_graph, baseline_num_partitions)
        print("  ✅ METIS完成")
    except Exception as e:
        print(f"  ❌ METIS失败: {e}")
    
    # 尝试加载强化学习结果
    print("\n🤖 尝试加载强化学习结果...")
    rl_models_found = False
    
    # 查找可能的模型文件
    possible_model_paths = [
        "results/20250703_214514num=2/models/dqn_model_10nodes_2parts.pt",
        f"results/20250703_214514num=2/models/dqn_model_{graph.number_of_nodes()}nodes_{num_partitions}parts.pt",
        "results/20250703_214514num=2/models/gnn_dqn_model_10nodes_2parts.pt",
        f"results/20250703_214514num=2/models/gnn_dqn_model_{graph.number_of_nodes()}nodes_{num_partitions}parts.pt",
        "results/20250703_214514num=2/models/ppo_model_10nodes_2parts.pt",
        f"results/20250703_214514num=2/models/ppo_model_{graph.number_of_nodes()}nodes_{num_partitions}parts.pt",
        "results/20250703_214514num=2/models/gnn_ppo_model_10nodes_2parts.pt",
        f"results/20250703_214514num=2/models/gnn_ppo_model_{graph.number_of_nodes()}nodes_{num_partitions}parts.pt"
    ]
    
    print(f"  🔍 查找模型路径:")
    for path in possible_model_paths:
        exists = os.path.exists(path)
        print(f"    {path}: {'✅' if exists else '❌'}")
    
    # === 新增：图尺寸适配逻辑 ===
    original_graph = graph
    original_num_partitions = num_partitions
    adapted_graph = None
    adapted_num_partitions = None
    
    # --- 修正 NameError：总是在循环外定义 baseline_graph ---
    baseline_graph = original_graph
    baseline_num_partitions = original_num_partitions

    for model_path in possible_model_paths:
        if os.path.exists(model_path):
            try:
                filename = os.path.basename(model_path)
                
                # --- 修正：更完善的Agent选择逻辑 ---
                is_gnn = 'gnn' in filename.lower()
                is_ppo = 'ppo' in filename.lower()
                
                agent_type = "Unknown"
                if is_ppo:
                    agent_type = "GNN-PPO" if is_gnn else "PPO"
                elif 'dqn' in filename.lower():
                    agent_type = "GNN-DQN" if is_gnn else "DQN"
                
                if agent_type == "Unknown":
                    continue

                # --- PPO Agent检查 ---
                if is_ppo and not ppo_available:
                    print(f"  ⏭️  检测到PPO模型 {filename} 但PPO Agent不可用，已跳过。")
                    continue
                
                import re
                match = re.search(r'(\d+)nodes_(\d+)parts', filename)
                if not match:
                    print(f"  ⚠️ 无法从 {filename} 解析尺寸，已跳过")
                    continue
                    
                expected_nodes = int(match.group(1))
                expected_partitions = int(match.group(2))
                
                print(f"\n  ▶️ 发现 {agent_type} 模型: {filename}")
                print(f"    模型期望: {expected_nodes} 节点, {expected_partitions} 分区")
                print(f"    当前图:   {original_graph.number_of_nodes()} 节点, {original_num_partitions} 分区")
                
                current_graph = original_graph
                current_num_partitions = original_num_partitions
                
                # 尺寸适配逻辑...
                if expected_nodes != current_graph.number_of_nodes():
                    if expected_nodes < current_graph.number_of_nodes():
                        print(f"    🔧 尺寸不匹配，为模型创建 {expected_nodes} 节点的子图...")
                        nodes_with_weights = sorted(original_graph.nodes(data='weight', default=1.0), key=lambda x: x[1], reverse=True)
                        selected_nodes = [node for node, _ in nodes_with_weights[:expected_nodes]]
                        subgraph = original_graph.subgraph(selected_nodes).copy()
                        node_mapping = {old_node: i for i, old_node in enumerate(subgraph.nodes())}
                        subgraph = nx.relabel_nodes(subgraph, node_mapping)
                        if not nx.is_connected(subgraph):
                            components = list(nx.connected_components(subgraph))
                            for i in range(len(components) - 1):
                                subgraph.add_edge(list(components[i])[0], list(components[i+1])[0])
                        current_graph = subgraph
                        current_num_partitions = expected_partitions
                        adapted_graph = current_graph # 记录适配后的图
                        adapted_num_partitions = current_num_partitions
                        print(f"    ✅ 子图创建成功: {current_graph.number_of_nodes()} 节点, {current_num_partitions} 分区")
                    else:
                        print(f"    ❌ 模型期望的图尺寸 ({expected_nodes}) 大于当前图 ({current_graph.number_of_nodes()})，无法适配。")
                        continue
                else:
                    print(f"    ✅ 尺寸匹配，直接使用当前图。")
                    current_num_partitions = expected_partitions
                
                env = GraphPartitionEnvironment(current_graph, current_num_partitions)
                agent = None
                agent_name = agent_type

                # --- 修正：根据Agent类型和错误日志提供正确配置 ---
                if agent_type == "DQN":
                    config = {'hidden_sizes': [512, 256]} 
                    state_size = len(current_graph.nodes()) * (current_num_partitions + 2)
                    action_size = len(current_graph.nodes()) * current_num_partitions
                    agent = DQNAgent(state_size, action_size, config=config)
                
                elif agent_type == "GNN-DQN":
                    config = {'hidden_dim': 64, 'num_layers': 4}
                    agent = GNNDQNAgent(current_graph, current_num_partitions, config=config)
                
                elif agent_type == "PPO":
                    # 假设PPOAgent与DQNAgent有相似的配置结构
                    config = {'hidden_sizes': [256, 256]} # 这是基于PPO错误日志的猜测
                    state_size = len(current_graph.nodes()) * (current_num_partitions + 2)
                    action_size = len(current_graph.nodes()) * current_num_partitions
                    agent = PPOAgent(state_size, action_size, config=config)
                
                elif agent_type == "GNN-PPO":
                    # --- 修复：为GNN-PPO提供正确的配置和初始化参数 ---
                    # GNN Agent的state_size是节点特征的维度
                    state_size = current_num_partitions + 2 
                    action_size = len(current_graph.nodes()) * current_num_partitions
                    config = {
                        'num_partitions': current_num_partitions,
                        'hidden_dim': 2048  # 匹配训练时使用的模型尺寸
                    }
                    agent = GNNPPOAgent(state_size, action_size, config=config)
                
                if agent:
                    agent.load_model(model_path)
                    state, _ = env.reset()
                    done = False
                    step_count = 0
                    max_steps = 200
                    while not done and step_count < max_steps:
                        # --- 修复：为GNN Agent提供正确的图数据结构 ---
                        if agent_type in ["GNN-DQN", "GNN-PPO"]:
                            # GNN Agent需要结构化的图数据，而不是扁平化的state
                            graph_data_for_act = env.get_state(format='graph')
                            action = agent.act(graph_data_for_act)
                        else:
                            action = agent.act(state)
                        
                        state, _, done, _, _ = env.step(action)
                        step_count += 1
                    
                    partitions[agent_name] = env.partition_assignment.copy()
                    print(f"  ✅ {agent_name} 模型加载并执行成功: {model_path}")
                    if adapted_graph:
                        print(f"      📝 注意: 使用了 {adapted_graph.number_of_nodes()} 节点的子图进行评估")
                    rl_models_found = True
            
            except Exception as e:
                import traceback
                print(f"  ❌ 加载模型 {model_path} 失败: {e}")
                # traceback.print_exc() # 可选：打印更详细的堆栈信息

    # === 修复：确保所有分区结果格式一致 ===
    print(f"\n🔧 检查分区结果格式...")
    valid_partitions = {}
    
    for name, partition in partitions.items():
        try:
            # 确保分区是字典格式 {node_id: partition_id}
            if isinstance(partition, dict):
                # 已经是字典格式，检查键是否为图中的节点
                if all(node in graph.nodes() for node in partition.keys()):
                    valid_partitions[name] = partition
                    print(f"  ✅ {name}: 字典格式，包含 {len(partition)} 个节点")
                else:
                    # --- 修复：处理DQN/GNN使用子图导致节点不匹配的问题 ---
                    if name in ["DQN", "GNN-DQN", "PPO", "GNN-PPO"] and adapted_graph is not None:
                         # 这是预期的行为，因为RL模型可能在子图上运行
                        print(f"  - {name}: 在 {len(partition)} 节点的子图上运行，结果有效。")
                        # 为了可视化，我们需要将子图分区映射回原始图
                        # 注意：这只是为了可视化，实际评估应分开
                        # 这里我们只保留它在子图上的划分结果，在画图时单独处理
                        valid_partitions[name] = partition
                    else:
                        print(f"  ❌ {name}: 字典格式但节点ID与主图不匹配")
            
            elif isinstance(partition, (list, np.ndarray)):
                # --- 修正：使用正确的图进行检查 ---
                current_graph_for_check = baseline_graph
                # 如果是RL agent且使用了子图，应使用子图的节点数检查
                if name in ["DQN", "GNN-DQN", "PPO", "GNN-PPO"] and adapted_graph is not None:
                    current_graph_for_check = adapted_graph
                
                # 数组格式，转换为字典
                if len(partition) == current_graph_for_check.number_of_nodes():
                    partition_dict = {node: int(partition[i]) 
                                      for i, node in enumerate(current_graph_for_check.nodes())}
                    valid_partitions[name] = partition_dict
                    print(f"  ✅ {name}: 数组格式转换为字典，包含 {len(partition_dict)} 个节点")
                else:
                    print(f"  ❌ {name}: 数组长度 {len(partition)} 不匹配节点数 {current_graph_for_check.number_of_nodes()}")
            
            else:
                print(f"  ❌ {name}: 未知格式 {type(partition)}")
                
        except Exception as e:
            print(f"  ❌ {name}: 处理时出错 {e}")
    
    partitions = valid_partitions
    
    # 创建可视化
    print(f"\n🎨 生成可视化结果...")
    num_methods = len(partitions)
    cols = 3
    rows = (num_methods + cols - 1) // cols  # 向上取整
    
    plt.figure(figsize=layout_params['figsize'])
    
    # --- 修复：为RL和基线算法准备不同的图和布局 ---
    baseline_pos = create_geographic_layout(baseline_graph)
    if adapted_graph:
        rl_pos = create_geographic_layout(adapted_graph)
    else:
        rl_pos = baseline_pos

    for i, (name, partition) in enumerate(partitions.items(), 1):
        plt.subplot(rows, cols, i)

        # --- 选择正确的图、布局和分区数进行可视化 ---
        if name in ["DQN", "GNN-DQN", "PPO", "GNN-PPO"] and adapted_graph is not None:
            current_graph_for_plot = adapted_graph
            current_pos = rl_pos
            current_num_partitions_for_plot = adapted_num_partitions
        else:
            current_graph_for_plot = baseline_graph
            current_pos = baseline_pos
            current_num_partitions_for_plot = baseline_num_partitions
        
        # 为每个分区分配不同的颜色
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink']
        
        # === 修复：正确处理节点颜色映射 ===
        node_colors = []
        for node in current_graph_for_plot.nodes():
            if node in partition:
                part_id = partition[node]
                node_colors.append(colors[part_id % len(colors)])
            else:
                node_colors.append('gray')  # 未分配的节点用灰色
        
        # 绘制节点和边
        nx.draw_networkx_nodes(current_graph_for_plot, current_pos, node_color=node_colors, 
                              node_size=layout_params['node_size'], alpha=0.8)
        nx.draw_networkx_edges(current_graph_for_plot, current_pos, alpha=0.5, width=0.5)
        
        # 根据图的大小决定是否显示节点标签
        if current_graph_for_plot.number_of_nodes() <= 30:
            if is_real_graph:
                # 真实图只显示节点ID
                node_labels = {i: str(i) for i in current_graph_for_plot.nodes()}
            else:
                # 测试图显示ID和权重
                node_labels = {i: f"{i}:{current_graph_for_plot.nodes[i]['weight']}" for i in current_graph_for_plot.nodes()}
            
            nx.draw_networkx_labels(current_graph_for_plot, current_pos, labels=node_labels, 
                                  font_size=layout_params['font_size'])
        
        # === 修复：正确计算分区统计信息 ===
        partition_weights = [0.0] * current_num_partitions_for_plot
        partition_counts = [0] * current_num_partitions_for_plot
        
        for node in current_graph_for_plot.nodes():
            if node in partition:
                part_id = partition[node]
                if 0 <= part_id < current_num_partitions_for_plot:  # 确保分区ID有效
                    partition_weights[part_id] += current_graph_for_plot.nodes[node].get('weight', 1.0)
                    partition_counts[part_id] += 1
        
        # 计算权重方差
        weight_variance = np.var(partition_weights) if partition_weights else 0.0
        
        plt.title(f"{name}\nWeights: {[f'{w:.1f}' for w in partition_weights]}\n"
                 f"Nodes: {partition_counts}\nVariance: {weight_variance:.2f}", 
                 fontsize=layout_params['font_size'])
        plt.axis('off')
    
    plt.suptitle(f"Algorithm Partition Comparison - {graph_name}", fontsize=14)
    plt.tight_layout()
    
    # 保存结果
    os.makedirs("results/plots", exist_ok=True)
    output_filename = f"all_partitions_comparison_{'real' if is_real_graph else 'test'}.png"
    plt.savefig(f"results/plots/{output_filename}", dpi=300, bbox_inches='tight')
    print(f"📁 可视化结果已保存: results/plots/{output_filename}")
    
    plt.show()
    
    # 打印总结
    print(f"\n📋 划分结果总结:")
    print(f"图类型: {graph_name}")
    print(f"分区数: {num_partitions}")
    print(f"算法数量: {len(partitions)}")
    for name, partition in partitions.items():
        # --- 修复：为RL和基线算法选择正确的图和分区数进行总结 ---
        if name in ["DQN", "GNN-DQN", "PPO", "GNN-PPO"] and adapted_graph is not None:
            current_graph_for_summary = adapted_graph
            current_num_partitions_for_summary = adapted_num_partitions
        else:
            current_graph_for_summary = baseline_graph
            current_num_partitions_for_summary = baseline_num_partitions

        partition_weights = [0.0] * current_num_partitions_for_summary
        for node in current_graph_for_summary.nodes():
            if node in partition:
                part_id = partition[node]
                if 0 <= part_id < current_num_partitions_for_summary:
                    partition_weights[part_id] += current_graph_for_summary.nodes[node].get('weight', 1.0)
        
        variance = np.var(partition_weights) if partition_weights else 0.0
        max_weight = max(partition_weights) if partition_weights else 0.0
        min_weight = min(p for p in partition_weights if p > 0) if any(p > 0 for p in partition_weights) else 0.0
        balance_ratio = max_weight / min_weight if min_weight > 0 else float('inf')
        
        print(f"  {name}: 权重方差={variance:.2f}, 平衡比={balance_ratio:.2f}")

if __name__ == "__main__":
    main()