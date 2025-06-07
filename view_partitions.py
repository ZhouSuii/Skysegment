# view_partitions.py -- 对比各个算法的划分结果
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from new_environment import GraphPartitionEnvironment  # 使用更新的环境
from baselines import random_partition, weighted_greedy_partition, spectral_partition, metis_partition
from run_experiments import create_test_graph

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
    num_partitions = 3 if graph.number_of_nodes() > 15 else 2
    print(f"📊 使用 {num_partitions} 个分区进行划分")
    
    # 获取布局参数
    layout_params = calculate_layout_parameters(graph, is_real_graph)
    
    # 获取不同算法的划分结果
    print("\n🧮 计算各算法的划分结果...")
    partitions = {}
    
    try:
        partitions["Random"] = random_partition(graph, num_partitions)
        print("  ✅ Random完成")
    except Exception as e:
        print(f"  ❌ Random失败: {e}")
    
    try:
        partitions["Greedy"] = weighted_greedy_partition(graph, num_partitions)
        print("  ✅ Greedy完成")
    except Exception as e:
        print(f"  ❌ Greedy失败: {e}")
    
    try:
        partitions["Spectral"] = spectral_partition(graph, num_partitions)
        print("  ✅ Spectral完成")
    except Exception as e:
        print(f"  ❌ Spectral失败: {e}")
    
    try:
        partitions["METIS"] = metis_partition(graph, num_partitions)
        print("  ✅ METIS完成")
    except Exception as e:
        print(f"  ❌ METIS失败: {e}")
    
    # 尝试加载强化学习结果
    print("\n🤖 尝试加载强化学习结果...")
    rl_models_found = False
    
    # 查找可能的模型文件
    possible_model_paths = [
        "results/models/dqn_model_10nodes_2parts.pt",
        f"results/models/dqn_model_{graph.number_of_nodes()}nodes_{num_partitions}parts.pt",
        "results/models/ppo_model_10nodes_2parts.pt",
        f"results/models/ppo_model_{graph.number_of_nodes()}nodes_{num_partitions}parts.pt",
        "results/models/gnn_ppo_model_10nodes_2parts.pt",
        f"results/models/gnn_ppo_model_{graph.number_of_nodes()}nodes_{num_partitions}parts.pt"
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
    
    for model_path in possible_model_paths:
        if os.path.exists(model_path):
            try:
                # 从文件名提取模型期望的图尺寸
                filename = os.path.basename(model_path)
                if 'dqn' in model_path:
                    # 解析文件名获取期望的节点数和分区数
                    import re
                    match = re.search(r'(\d+)nodes_(\d+)parts', filename)
                    if match:
                        expected_nodes = int(match.group(1))
                        expected_partitions = int(match.group(2))
                        
                        print(f"  🔍 模型 {filename} 期望: {expected_nodes}节点, {expected_partitions}分区")
                        print(f"      当前图: {original_graph.number_of_nodes()}节点, {original_num_partitions}分区")
                        
                        # 如果尺寸匹配，直接使用
                        if (expected_nodes == original_graph.number_of_nodes() and 
                            expected_partitions == original_num_partitions):
                            graph = original_graph
                            num_partitions = original_num_partitions
                            print(f"      ✅ 尺寸匹配，直接使用")
                        
                        # 如果尺寸不匹配，尝试创建子图
                        elif expected_nodes < original_graph.number_of_nodes():
                            print(f"      🔧 尺寸不匹配，创建{expected_nodes}节点的子图...")
                            
                            # 创建子图（选择权重最大的节点）
                            nodes_with_weights = [(node, original_graph.nodes[node]['weight']) 
                                                for node in original_graph.nodes()]
                            nodes_with_weights.sort(key=lambda x: x[1], reverse=True)
                            selected_nodes = [node for node, _ in nodes_with_weights[:expected_nodes]]
                            
                            # 创建子图并重新编号
                            subgraph = original_graph.subgraph(selected_nodes).copy()
                            node_mapping = {old_node: i for i, old_node in enumerate(subgraph.nodes())}
                            subgraph = nx.relabel_nodes(subgraph, node_mapping)
                            
                            # 确保子图连通
                            if not nx.is_connected(subgraph):
                                # 添加边确保连通
                                components = list(nx.connected_components(subgraph))
                                for i in range(len(components) - 1):
                                    node1 = list(components[i])[0]
                                    node2 = list(components[i+1])[0]
                                    subgraph.add_edge(node1, node2)
                            
                            graph = subgraph
                            num_partitions = expected_partitions
                            adapted_graph = graph
                            adapted_num_partitions = num_partitions
                            
                            print(f"      ✅ 子图创建成功: {graph.number_of_nodes()}节点, {graph.number_of_edges()}边")
                        
                        else:
                            print(f"      ❌ 模型期望的图太大，无法适配")
                            continue
                
                    # 加载DQN结果
                    from agent_dqn_basic import DQNAgent
                    env = GraphPartitionEnvironment(graph, num_partitions)
                    state_size = len(graph.nodes()) * (num_partitions + 2)  # 更新状态大小
                    action_size = len(graph.nodes()) * num_partitions
                    agent = DQNAgent(state_size, action_size)
                    agent.load_model(model_path)
                    
                    state, _ = env.reset()
                    done = False
                    step_count = 0
                    max_steps = 200
                    while not done and step_count < max_steps:
                        action = agent.act(state)
                        state, _, done, _, _ = env.step(action)
                        step_count += 1
                    
                    partitions["DQN"] = env.partition_assignment.copy()
                    print(f"  ✅ DQN模型加载成功: {model_path}")
                    if adapted_graph is not None:
                        print(f"      📝 注意: 使用了{adapted_graph.number_of_nodes()}节点的子图")
                    rl_models_found = True
                    break
                    
            except Exception as e:
                print(f"  ⚠️ 加载模型失败 {model_path}: {e}")
    
    # 恢复原始图用于基线算法比较（如果使用了子图）
    if adapted_graph is not None:
        print(f"\n📝 提示: 强化学习使用了{adapted_graph.number_of_nodes()}节点子图，基线算法使用原始{original_graph.number_of_nodes()}节点图")
        baseline_graph = original_graph
        baseline_num_partitions = original_num_partitions
    else:
        baseline_graph = graph
        baseline_num_partitions = num_partitions
    
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
                    print(f"  ❌ {name}: 字典格式但节点ID不匹配")
            
            elif isinstance(partition, (list, np.ndarray)):
                # 数组格式，转换为字典
                if len(partition) == graph.number_of_nodes():
                    partition_dict = {node: int(partition[i]) for i, node in enumerate(graph.nodes())}
                    valid_partitions[name] = partition_dict
                    print(f"  ✅ {name}: 数组格式转换为字典，包含 {len(partition_dict)} 个节点")
                else:
                    print(f"  ❌ {name}: 数组长度 {len(partition)} 不匹配节点数 {graph.number_of_nodes()}")
            
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
    
    # 创建统一的布局
    pos = create_geographic_layout(graph)
    
    for i, (name, partition) in enumerate(partitions.items(), 1):
        plt.subplot(rows, cols, i)
        
        # 为每个分区分配不同的颜色
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink']
        
        # === 修复：正确处理节点颜色映射 ===
        node_colors = []
        for node in graph.nodes():
            if node in partition:
                part_id = partition[node]
                node_colors.append(colors[part_id % len(colors)])
            else:
                node_colors.append('gray')  # 未分配的节点用灰色
        
        # 绘制节点和边
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=layout_params['node_size'], alpha=0.8)
        nx.draw_networkx_edges(graph, pos, alpha=0.5, width=0.5)
        
        # 根据图的大小决定是否显示节点标签
        if graph.number_of_nodes() <= 30:
            if is_real_graph:
                # 真实图只显示节点ID
                node_labels = {i: str(i) for i in graph.nodes()}
            else:
                # 测试图显示ID和权重
                node_labels = {i: f"{i}:{graph.nodes[i]['weight']}" for i in graph.nodes()}
            
            nx.draw_networkx_labels(graph, pos, labels=node_labels, 
                                  font_size=layout_params['font_size'])
        
        # === 修复：正确计算分区统计信息 ===
        partition_weights = [0.0] * num_partitions
        partition_counts = [0] * num_partitions
        
        for node in graph.nodes():
            if node in partition:
                part_id = partition[node]
                if 0 <= part_id < num_partitions:  # 确保分区ID有效
                    partition_weights[part_id] += graph.nodes[node]['weight']
                    partition_counts[part_id] += 1
        
        # 计算权重方差
        weight_variance = np.var(partition_weights) if partition_weights else 0.0
        
        plt.title(f"{name}\n权重: {[f'{w:.1f}' for w in partition_weights]}\n"
                 f"节点数: {partition_counts}\n方差: {weight_variance:.2f}", 
                 fontsize=layout_params['font_size'])
        plt.axis('off')
    
    plt.suptitle(f"算法划分结果对比 - {graph_name}", fontsize=14)
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
        partition_weights = [0.0] * num_partitions
        for node in graph.nodes():
            if node in partition:
                part_id = partition[node]
                if 0 <= part_id < num_partitions:
                    partition_weights[part_id] += graph.nodes[node]['weight']
        
        variance = np.var(partition_weights) if partition_weights else 0.0
        max_weight = max(partition_weights) if partition_weights else 0.0
        min_weight = min(p for p in partition_weights if p > 0) if any(p > 0 for p in partition_weights) else 0.0
        balance_ratio = max_weight / min_weight if min_weight > 0 else float('inf')
        
        print(f"  {name}: 权重方差={variance:.2f}, 平衡比={balance_ratio:.2f}")

if __name__ == "__main__":
    main()