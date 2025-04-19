# view_partitions.py -- 对比各个算法的划分结果
import networkx as nx
import matplotlib.pyplot as plt
from environment import GraphPartitionEnvironment
from baselines import random_partition, weighted_greedy_partition, spectral_partition, metis_partition
from run_experiments import create_test_graph

# 创建测试图
graph = create_test_graph(num_nodes=10, seed=42)
num_partitions = 2

# 获取不同算法的划分结果
partitions = {
    "Random": random_partition(graph, num_partitions),
    "Greedy": weighted_greedy_partition(graph, num_partitions),
    "Spectral": spectral_partition(graph, num_partitions),
    "METIS": metis_partition(graph, num_partitions)
}

# 加载DQN结果（从结果CSV读取）
import pandas as pd

try:
    df = pd.read_csv("results/test_graph_10_results.csv")
    print("找到结果文件，加载DQN模型...")
    from agent_dqn_basic import DQNAgent

    # 从保存的模型中加载DQN结果
    env = GraphPartitionEnvironment(graph, num_partitions)
    state_size = len(graph.nodes()) * (num_partitions + 1)
    action_size = len(graph.nodes()) * num_partitions
    agent = DQNAgent(state_size, action_size)
    agent.load_model("results/models/dqn_model_10nodes_2parts.pt")

    # 执行一次完整的划分
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, _, done, _, _ = env.step(action)

    partitions["DQN"] = env.partition_assignment
except:
    print("无法加载DQN结果，只显示基线算法")

# 显示所有划分结果
plt.figure(figsize=(15, 10))
for i, (name, partition) in enumerate(partitions.items(), 1):
    plt.subplot(2, 3, i)

    env = GraphPartitionEnvironment(graph, num_partitions)
    env.partition_assignment = partition

    # 为每个分区分配不同的颜色
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    node_colors = [colors[env.partition_assignment[i] % len(colors)] for i in range(env.num_nodes)]

    # 获取节点位置
    pos = nx.spring_layout(graph, seed=42)

    # 绘制节点和边
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(graph, pos)

    # 显示节点权重
    node_labels = {i: f"{i}:{graph.nodes[i]['weight']}" for i in range(env.num_nodes)}
    nx.draw_networkx_labels(graph, pos, labels=node_labels)

    # 计算每个分区的总权重
    partition_weights = [0] * num_partitions
    for i in range(env.num_nodes):
        partition_weights[env.partition_assignment[i]] += graph.nodes[i]['weight']

    plt.title(f"{name} - 分区权重: {partition_weights}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("results/plots/all_partitions_comparison.png")
plt.show()