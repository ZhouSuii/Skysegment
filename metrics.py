# evaluate the model
import numpy as np
import networkx as nx
from collections import defaultdict


def calculate_partition_weights(graph, partition_assignment, num_partitions=None):
    """
    计算每个分区的总权重

    参数:
        graph: 输入图对象
        partition_assignment: 节点分区分配数组
        num_partitions: 分区数量，如果为None则从partition_assignment推断

    返回:
        array: 每个分区的总权重
    """
    if num_partitions is None:
        num_partitions = max(partition_assignment) + 1

    partition_weights = np.zeros(num_partitions)

    for i in range(len(partition_assignment)):
        partition = partition_assignment[i]
        node_weight = graph.nodes[i].get('weight', 1)  # 如果没有权重，默认为1
        partition_weights[partition] += node_weight

    return partition_weights


def calculate_weight_variance(graph, partition_assignment, num_partitions=None):
    """
    计算分区权重方差

    参数:
        graph: 输入图对象
        partition_assignment: 节点分区分配数组
        num_partitions: 分区数量，如果为None则从partition_assignment推断

    返回:
        float: 分区权重方差
    """
    partition_weights = calculate_partition_weights(graph, partition_assignment, num_partitions)
    return np.var(partition_weights)


def calculate_weight_imbalance(graph, partition_assignment, num_partitions=None):
    """
    计算权重不平衡度（最大权重与最小权重的比值）

    参数:
        graph: 输入图对象
        partition_assignment: 节点分区分配数组
        num_partitions: 分区数量，如果为None则从partition_assignment推断

    返回:
        float: 权重不平衡度
    """
    partition_weights = calculate_partition_weights(graph, partition_assignment, num_partitions)
    max_weight = max(partition_weights)
    min_weight = min(partition_weights) if min(partition_weights) > 0 else 1
    return max_weight / min_weight


def calculate_edge_cut(graph, partition_assignment):
    """
    计算切割边的数量

    参数:
        graph: 输入图对象
        partition_assignment: 节点分区分配数组

    返回:
        int: 切割边的数量
    """
    cut_edges = 0
    for u, v in graph.edges():
        if partition_assignment[u] != partition_assignment[v]:
            cut_edges += 1

    return cut_edges


def calculate_normalized_cut(graph, partition_assignment):
    """
    计算归一化的切割边比例

    参数:
        graph: 输入图对象
        partition_assignment: 节点分区分配数组

    返回:
        float: 归一化的切割边比例
    """
    cut_edges = calculate_edge_cut(graph, partition_assignment)
    return cut_edges / max(1, len(graph.edges()))


def calculate_modularity(graph, partition_assignment, num_partitions=None):
    """
    计算图划分的模块度

    参数:
        graph: 输入图对象
        partition_assignment: 节点分区分配数组
        num_partitions: 分区数量，如果为None则从partition_assignment推断

    返回:
        float: 模块度 Q
    """
    if num_partitions is None:
        num_partitions = max(partition_assignment) + 1

    m = len(graph.edges())
    if m == 0:
        return 0

    # 构建社区
    communities = defaultdict(list)
    for node, part in enumerate(partition_assignment):
        communities[part].append(node)
    communities = list(communities.values())

    q = 0
    for community in communities:
        for i in community:
            for j in community:
                if i < j:  # 避免重复计算
                    if graph.has_edge(i, j):
                        actual = 1
                    else:
                        actual = 0
                    expected = graph.degree(i) * graph.degree(j) / (2 * m)
                    q += (actual - expected)

    return q / m


def evaluate_partition(graph, partition_assignment, num_partitions=None, print_results=True):
    """
    全面评估一个图划分结果

    参数:
        graph: 输入图对象
        partition_assignment: 节点分区分配数组
        num_partitions: 分区数量，如果为None则从partition_assignment推断
        print_results: 是否打印评估结果

    返回:
        dict: 包含各评估指标的字典
    """
    if num_partitions is None:
        num_partitions = max(partition_assignment) + 1

    partition_weights = calculate_partition_weights(graph, partition_assignment, num_partitions)
    weight_variance = calculate_weight_variance(graph, partition_assignment, num_partitions)
    weight_imbalance = calculate_weight_imbalance(graph, partition_assignment, num_partitions)
    edge_cut = calculate_edge_cut(graph, partition_assignment)
    normalized_cut = calculate_normalized_cut(graph, partition_assignment)
    modularity = calculate_modularity(graph, partition_assignment, num_partitions)

    results = {
        "partition_weights": partition_weights,
        "weight_variance": weight_variance,
        "weight_imbalance": weight_imbalance,
        "edge_cut": edge_cut,
        "normalized_cut": normalized_cut,
        "modularity": modularity
    }

    if print_results:
        print(f"分区权重: {partition_weights}")
        print(f"权重方差: {weight_variance:.4f}")
        print(f"权重不平衡度: {weight_imbalance:.4f}")
        print(f"切割边数量: {edge_cut}")
        print(f"归一化切割: {normalized_cut:.4f}")
        print(f"模块度Q: {modularity:.4f}")

    return results