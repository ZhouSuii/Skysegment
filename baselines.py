# random partitioning and metis
import numpy as np
import networkx as nx
import random

try:
    import pymetis
    METIS_AVAILABLE = True
except ImportError:
    METIS_AVAILABLE = False
    print("警告: METIS库未安装，无法使用METIS算法。使用pip install metis安装。")


def random_partition(graph, num_partitions=2):
    """
    随机划分图

    参数:
        graph: 输入图对象
        num_partitions: 分区数量

    返回:
        array: 节点分区分配数组
    """
    num_nodes = len(graph.nodes())
    partition_assignment = np.random.randint(0, num_partitions, num_nodes)

    # 确保每个分区至少有一个节点
    for p in range(num_partitions):
        if p not in partition_assignment:
            # 随机选择一个节点分配到该分区
            random_node = random.randint(0, num_nodes - 1)
            partition_assignment[random_node] = p

    return partition_assignment


def weighted_greedy_partition(graph, num_partitions=2):
    """
    基于节点权重的贪心划分算法

    参数:
        graph: 输入图对象
        num_partitions: 分区数量

    返回:
        array: 节点分区分配数组
    """
    num_nodes = len(graph.nodes())

    # 获取所有节点及其权重
    nodes_with_weights = [(i, graph.nodes[i].get('weight', 1)) for i in range(num_nodes)]

    # 按权重降序排序
    nodes_with_weights.sort(key=lambda x: x[1], reverse=True)

    # 初始化每个分区的总权重
    partition_weights = np.zeros(num_partitions)
    partition_assignment = np.zeros(num_nodes, dtype=int)

    # 逐个分配节点
    for node, weight in nodes_with_weights:
        # 找到当前权重最小的分区
        target_partition = np.argmin(partition_weights)
        partition_assignment[node] = target_partition
        partition_weights[target_partition] += weight

    return partition_assignment


def metis_partition(graph, num_partitions=2):
    """
    使用METIS库进行图划分

    参数:
        graph: 输入图对象
        num_partitions: 分区数量

    返回:
        array: 节点分区分配数组
    """
    if not METIS_AVAILABLE:
        print("METIS库不可用，使用随机划分代替")
        return random_partition(graph, num_partitions)

    # 转换图到PyMETIS可接受的格式
    adjacency_list = []
    for i in range(len(graph.nodes())):
        adjacency_list.append(list(graph.neighbors(i)))

    # 获取节点权重
    node_weights = [graph.nodes[i].get('weight', 1) for i in range(len(graph.nodes()))]

    try:
        # PyMETIS直接支持节点权重
        _, partition_assignment = pymetis.part_graph(num_partitions, adjacency=adjacency_list,
                                                     vweights=node_weights)
        return np.array(partition_assignment)
    except Exception as e:
        print(f"METIS算法出错: {e}，使用随机划分代替")
        return random_partition(graph, num_partitions)


def spectral_partition(graph, num_partitions=2):
    """
    使用谱聚类进行图划分

    参数:
        graph: 输入图对象
        num_partitions: 分区数量

    返回:
        array: 节点分区分配数组
    """
    # 获取拉普拉斯矩阵
    laplacian = nx.normalized_laplacian_matrix(graph)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(laplacian.toarray())

    # 按特征值排序
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 使用前k个非零特征向量
    k = num_partitions
    features = eigenvectors[:, 1:k + 1]  # 跳过第一个特征向量(对应特征值0)

    # 使用k-means进行聚类
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_partitions, random_state=42)
    partition_assignment = kmeans.fit_predict(features)

    return partition_assignment