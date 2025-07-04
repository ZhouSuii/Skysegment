import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, atan2, degrees
import time
import networkx as nx
from scipy.spatial import ConvexHull, KDTree, Delaunay # 导入 KDTree 和 Delaunay
import matplotlib.patches as patches
import traceback
from sklearn.cluster import KMeans # 导入 K-Means 聚类


# --- 配置参数 ---
# 使用我们之前处理好的 Parquet 文件
PARQUET_FILE_PATH = 'processed_data/all_flights_2024-11-11.parquet'
# 我们将动态决定节点数量，但保留这个变量用于后续步骤
NUM_GRAPH_NODES = 200 # 根据用户的选择，设置为200

# --- 新增配置 ---
# 是否运行肘部法则来寻找最佳节点数
# 设置为 True 来运行分析，设置为 False 并填入 NUM_GRAPH_NODES 来生成图
RUN_ELBOW_METHOD_ANALYSIS = False # 关闭分析模式，进入图生成阶段
ELBOW_K_RANGE = range(20, 501, 20) # 扩大并细化k值范围，步长为10
ELBOW_PLOT_OUTPUT_FILE = 'elbow_method_analysis2.png'


# 图节点生成参数
# NUM_GRAPH_NODES = 30 # 被上面的占位符取代

# 权重计算系数 (根据你文本公式中的 alpha 参数和 delta W 的含义)
# w_i = W_i,0 + alpha1 * DeltaW_path + alpha2 * DeltaW_turn + alpha3 * DeltaW_hspeed + alpha4 * DeltaW_vspeed
# 这里的 DeltaW_path/turn/hspeed/vspeed 指的是累加到节点 i 上的对应类型总增量
# DeltaW_path: 轨迹点经过节点 i 附近的次数
# DeltaW_turn: 轨迹点在节点 i 附近发生转弯的次数或转弯角度之和
# DeltaW_hspeed: 轨迹点在节点 i 附近发生水平速度变化的累积量 (例如，速度变化绝对值之和)
# DeltaW_vspeed: 轨迹点在节点 i 附近发生垂直速度变化的累积量 (例如，垂直速度绝对值之和)

W_INITIAL = 1.0 # 节点初始权重 W_i,0
A1_PATH = 0.2 # 轨迹点经过的权重系数
A2_TURN = 0.5 # 转弯权重系数
A3_HSPEED = 1.0 # 水平速度变化权重系数
A4_VSPEED = 1.0 # 垂直速度变化权重系数

TURN_THRESHOLD = 15 # 转弯阈值 (度)
# 速度变化阈值 - 根据实际数据分布调整，这里使用概念值
HSPEED_CHANGE_THRESHOLD = 10 # 水平速度变化阈值 (例如，节)
VSPEED_CHANGE_THRESHOLD = 100 # 垂直速度变化阈值 (例如，ft/min)
# 距离阈值 - 轨迹点在多远范围内算作"经过"节点 i
# 这里的单位是度，对于更广阔的区域，这个值可能需要小心调整
PROXIMITY_THRESHOLD_DEG = 0.05 # 距离阈值 (度)，比之前稍小以适应更大范围

# 输出文件名
GRAPH_OUTPUT_FILE = 'airspace_graph_full.graphml' # 修改文件名以反映新数据源
PLOT_OUTPUT_FILE = 'airspace_graph_full_visualization.png' # 修改文件名

# --- 辅助函数 ---
def calculate_heading_change(h1, h2):
    """计算两个航向 (0-360度) 之间的绝对差值，处理绕回情况"""
    if pd.isna(h1) or pd.isna(h2):
        return 0
    diff = abs(h1 - h2)
    return min(diff, 360 - diff)

def calculate_vspeed(alt1, alt2, t1, t2):
    """计算垂直速度 (ft/s)"""
    if pd.isna(alt1) or pd.isna(alt2) or pd.isna(t1) or pd.isna(t2):
        return 0
    delta_t = (t2 - t1).total_seconds()
    # 避免除以零或时间倒流
    if delta_t <= 1e-6:
        return 0
    delta_alt = alt2 - alt1
    # 将 ft/s 转换为 ft/min 以匹配 VSPEED_CHANGE_THRESHOLD 的单位 (如果需要)
    # return delta_alt / delta_t * 60 # 单位: ft/min
    return delta_alt / delta_t # 单位: ft/s


# --- 新增辅助函数: 肘部法则分析 ---
def find_optimal_nodes_elbow(coords, k_range, output_plot_file):
    """
    使用肘部法则来寻找K-Means的最佳k值 (节点数)。
    
    Args:
        coords (np.array): 用于聚类的坐标数据 (N, 2)。
        k_range (range): 要测试的k值的范围。
        output_plot_file (str): 肘部法则图的输出文件名。
    """
    print("\n--- Starting Elbow Method analysis to find the optimal number of nodes ---")
    inertias = []
    
    # --- 修改: 移除抽样，使用全部数据点 ---
    # if len(coords) > 200000: # 如果轨迹点超过20万
    #     print(f"数据量过大 ({len(coords)}), 随机抽取200,000个点进行肘部法则分析...")
    #     sample_indices = np.random.choice(coords.shape[0], 200000, replace=False)
    #     coords_sample = coords[sample_indices]
    # else:
    #     coords_sample = coords
    print(f"警告: 正在对全部 {len(coords)} 个数据点进行分析，这可能会非常耗时。")
    coords_sample = coords

    for k in k_range:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(coords_sample)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(12, 7))
    plt.plot(k_range, inertias, marker='o', linestyle='-')
    plt.title('Elbow Method for Optimal Number of Nodes')
    plt.xlabel('Number of Nodes (k)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.grid(True)
    plt.xticks(k_range)
    plt.savefig(output_plot_file)
    plt.close() # 关闭图像，防止在脚本末尾意外显示
    print(f"--- 肘部法则分析完成，图表已保存至: {output_plot_file} ---")


# --- 主逻辑 ---
print("Processing started...")
start_process_time = time.time()

# 1. 加载数据
try:
    print(f"正在加载 Parquet 文件: {PARQUET_FILE_PATH}")
    df = pd.read_parquet(PARQUET_FILE_PATH)
    print(f"原始数据行数: {len(df)}")

    # 转换时间戳列，强制错误转为 NaT
    # Parquet 通常能更好地保留类型，但以防万一
    if df['track_timestamp'].dtype == 'object':
        print("转换时间戳...")
        df['track_timestamp'] = pd.to_datetime(df['track_timestamp'], errors='coerce')

    # 统一航班标识：我们选择 'identification_callsign' 作为唯一标识符
    # 为避免冲突，直接丢弃 'identification_id'
    if 'identification_id' in df.columns:
        df.drop(columns=['identification_id'], inplace=True)

    # 定义需要检查的关键列，使用原始的列名
    required_cols = ['track_timestamp', 'identification_callsign', 'track_latitude', 'track_longitude',
                     'track_altitude', 'track_speed', 'track_heading']
    
    # 删除任何关键列为 NaT 或 NaN 的行
    print("清理无效数据...")
    initial_rows = len(df)
    df.dropna(subset=required_cols, inplace=True)
    print(f"因关键数据无效移除 {initial_rows - len(df)} 行")

    # 为了与脚本其余部分兼容，在清理之后再将 'identification_id' 重命名
    # if 'identification_id' in df.columns:
    #     df.rename(columns={'identification_id': 'identification_callsign'}, inplace=True)

    print(f"加载并清理后剩余有效数据点: {len(df)}")

except FileNotFoundError:
    print(f"错误：文件未找到 {PARQUET_FILE_PATH}")
    exit()
except Exception as e:
    print(f"读取或处理CSV时发生错误: {e}")
    traceback.print_exc()
    exit()

# 2. 按时间筛选和地理区域筛选
print("正在处理整个数据集，已跳过时间和地理区域筛选。")
df_filtered = df.copy()


if df_filtered.empty:
    print(f"在指定时间和地理区域内未找到数据。")
    exit()

print(f"筛选得到 {len(df_filtered)} 个有效轨迹点。")


# --- 新步骤: 运行肘部法则分析 (如果启用) ---
filtered_track_coords = df_filtered[['track_longitude', 'track_latitude']].values

if RUN_ELBOW_METHOD_ANALYSIS:
    find_optimal_nodes_elbow(filtered_track_coords, ELBOW_K_RANGE, ELBOW_PLOT_OUTPUT_FILE)
    print("\n肘部法则分析已完成。")
    print(f"请检查生成的图片 '{ELBOW_PLOT_OUTPUT_FILE}' 来确定一个理想的节点数量 'k'。")
    print("找到 'k' 后, 请修改脚本顶部的 'NUM_GRAPH_NODES' 为该值, 并将 'RUN_ELBOW_METHOD_ANALYSIS' 设置为 False, 然后重新运行脚本来生成最终的空域图。")
    exit() # 分析完成后退出，等待用户操作


# 3. 从过滤后的轨迹点生成图节点 (Original CD Points)
print(f"\n开始生成 {NUM_GRAPH_NODES} 个图节点 (Original CD Points)...")
# 提取所有过滤后的轨迹点的经纬度

if len(filtered_track_coords) < NUM_GRAPH_NODES:
    print(f"警告: 过滤后的轨迹点数量 ({len(filtered_track_coords)}) 少于所需的节点数量 ({NUM_GRAPH_NODES})。")
    # 使用所有轨迹点作为节点，但这可能导致节点过多且重叠
    graph_node_coords = filtered_track_coords
    num_graph_nodes_actual = len(graph_node_coords)
    print(f"使用所有过滤后的轨迹点作为节点 ({num_graph_nodes_actual} 个)。")
elif NUM_GRAPH_NODES <= 0:
     print("警告: 节点数量设置为非正数，无法生成节点。")
     graph_node_coords = np.array([])
     num_graph_nodes_actual = 0
else:
    # 使用 K-Means 聚类生成节点位置
    try:
        kmeans = KMeans(n_clusters=NUM_GRAPH_NODES, random_state=42, n_init=10) # Added n_init for stability
        kmeans.fit(filtered_track_coords)
        graph_node_coords = kmeans.cluster_centers_
        num_graph_nodes_actual = len(graph_node_coords)
        print(f"通过 K-Means 聚类生成了 {num_graph_nodes_actual} 个节点位置。")
    except Exception as e:
         print(f"K-Means 聚类出错: {e}")
         traceback.print_exc()
         graph_node_coords = np.array([])
         num_graph_nodes_actual = 0


if num_graph_nodes_actual == 0:
    print("无法生成图节点，退出处理。")
    exit()


# 4. 计算节点权重
print("\n开始计算节点权重...")
weight_calculation_start_time = time.time()

# 初始化节点权重和累加器
node_weights = np.full(num_graph_nodes_actual, W_INITIAL, dtype=np.float32)
node_point_counts = np.zeros(num_graph_nodes_actual, dtype=np.float32) # 累加 DeltaW_path (点的数量)
node_turn_contributions = np.zeros(num_graph_nodes_actual, dtype=np.float32) # 累加 DeltaW_turn
node_hspeed_contributions = np.zeros(num_graph_nodes_actual, dtype=np.float32) # 累加 DeltaW_hspeed
node_vspeed_contributions = np.zeros(num_graph_nodes_actual, dtype=np.float32) # 累加 DeltaW_vspeed


# 构建 KD-Tree 用于高效查找最近节点
# KDTree 使用的是欧氏距离，对于经纬度，在大区域会不准确，但对于小区域或作为近似是可接受的
# 更精确应使用地理距离或 UTM 坐标，这里为简化使用经纬度欧氏距离
node_kdtree = KDTree(graph_node_coords)

# 准备轨迹点数据，计算变化量
# 按航班排序
df_filtered = df_filtered.sort_values(['identification_callsign', 'track_timestamp'])
# 计算前后点的时间、位置、速度、航向差异
df_filtered['prev_timestamp'] = df_filtered.groupby('identification_callsign')['track_timestamp'].shift(1)
df_filtered['prev_latitude'] = df_filtered.groupby('identification_callsign')['track_latitude'].shift(1)
df_filtered['prev_longitude'] = df_filtered.groupby('identification_callsign')['track_longitude'].shift(1)
df_filtered['prev_altitude'] = df_filtered.groupby('identification_callsign')['track_altitude'].shift(1)
df_filtered['prev_speed'] = df_filtered.groupby('identification_callsign')['track_speed'].shift(1)
df_filtered['prev_heading'] = df_filtered.groupby('identification_callsign')['track_heading'].shift(1)

# 过滤掉每个航班的第一个点 (没有前一个点)
df_calc = df_filtered.dropna(subset=['prev_timestamp']).copy()

print(f"处理 {len(df_calc)} 个轨迹点以计算权重增量...")

# 计算各特征变化
df_calc['delta_t_secs'] = (df_calc['track_timestamp'] - df_calc['prev_timestamp']).dt.total_seconds()
# 过滤掉时间差小于等于零的点
df_calc = df_calc[df_calc['delta_t_secs'] > 1e-6].copy()
print(f"移除无效时间差后剩余 {len(df_calc)} 个点进行权重计算。")

df_calc['heading_change'] = df_calc.apply(lambda row: calculate_heading_change(row['prev_heading'], row['track_heading']), axis=1)
df_calc['is_turn'] = (df_calc['heading_change'] > TURN_THRESHOLD).astype(float) # 转向标志

df_calc['delta_hspeed'] = abs(df_calc['track_speed'] - df_calc['prev_speed']) # 水平速度变化量
# df_calc['is_hspeed_change'] = (df_calc['delta_hspeed'] > HSPEED_CHANGE_THRESHOLD).astype(float) # 水平速度变化标志

df_calc['vspeed_fps'] = df_calc.apply(lambda row: calculate_vspeed(row['prev_altitude'], row['track_altitude'], row['prev_timestamp'], row['track_timestamp']), axis=1)
# 将 ft/s 转换为 ft/min 用于阈值比较
df_calc['vspeed_ft_min'] = df_calc['vspeed_fps'] * 60
df_calc['delta_vspeed'] = abs(df_calc['vspeed_ft_min']) # 垂直速度变化量 (使用 ft/min)
# df_calc['is_vspeed_change'] = (df_calc['delta_vspeed'] > VSPEED_CHANGE_THRESHOLD).astype(float) # 垂直速度变化标志


# 遍历轨迹点，找到最近节点并累加贡献
processed_points_for_weight = 0
# 使用 values 迭代行以提高速度，虽然不及向量化，但对于非简单的累加是常用的折衷
for row_values in df_calc[['track_longitude', 'track_latitude', 'is_turn', 'delta_hspeed', 'delta_vspeed']].values:
    lon, lat, is_turn, delta_hspeed, delta_vspeed = row_values

    # 查询最近的节点索引
    # query 接受 (lon, lat)
    distance, nearest_node_idx = node_kdtree.query((lon, lat))

    # 如果轨迹点在节点附近 (距离小于阈值)，则累加贡献
    if distance < PROXIMITY_THRESHOLD_DEG:

        # 累加 DeltaW_path (轨迹点经过次数)
        node_point_counts[nearest_node_idx] += 1

        # 累加 DeltaW_turn (如果转弯，累加1)
        node_turn_contributions[nearest_node_idx] += is_turn # is_turn 是 0.0 或 1.0

        # 累加 DeltaW_hspeed (水平速度变化累积量)
        # 可以使用变化量本身，或超过阈值时累加固定值
        # node_hspeed_contributions[nearest_node_idx] += delta_hspeed # 累加变化量
        if delta_hspeed > HSPEED_CHANGE_THRESHOLD:
             node_hspeed_contributions[nearest_node_idx] += 1.0 # 超过阈值则计数

        # 累加 DeltaW_vspeed (垂直速度变化累积量)
        # node_vspeed_contributions[nearest_node_idx] += delta_vspeed # 累加变化量 (ft/min)
        if delta_vspeed > VSPEED_CHANGE_THRESHOLD:
            node_vspeed_contributions[nearest_node_idx] += 1.0 # 超过阈值则计数

        processed_points_for_weight += 1

print(f"Aggregated contributions from {processed_points_for_weight} trajectory points to their nearest nodes.")

# 计算最终节点权重 W_i
# W_i = W_i,0 + A1*Count + A2*Turn_Sum + A3*HSpeed_Sum + A4*VSpeed_Sum
# node_weights 在初始化时已经加上 W_INITIAL
node_weights += (A1_PATH * node_point_counts +
                 A2_TURN * node_turn_contributions +
                 A3_HSPEED * node_hspeed_contributions +
                 A4_VSPEED * node_vspeed_contributions)

# (可选) 对最终权重进行归一化，以便可视化时大小变化更明显
# 例如，缩放到 [1, Max_Marker_Size] 的范围
min_w = np.min(node_weights)
max_w = np.max(node_weights)
# 防止所有权重都相同
if max_w > min_w:
    normalized_weights = 10 + (node_weights - min_w) / (max_w - min_w) * 90 # 缩放到 10-100
else:
    normalized_weights = np.full_like(node_weights, 50) # 如果所有权重相同，使用固定大小

weight_calculation_time = time.time() - weight_calculation_start_time
print(f"节点权重计算耗时: {weight_calculation_time:.2f} 秒。")


# --- 5. 生成 NetworkX 图 ---
print("\n开始生成 NetworkX 图...")
graph_start_time = time.time()
G = nx.Graph()

# 添加节点及其属性 (位置, 权重)
# 使用整数索引作为节点 ID
for i in range(num_graph_nodes_actual):
     # graph_node_coords 存储的是 (lon, lat)
    G.add_node(i, weight=node_weights[i], lon=graph_node_coords[i, 0], lat=graph_node_coords[i, 1])

print(f"Added {G.number_of_nodes()} nodes to the graph.")

# 添加边 (使用 Delaunay 三角剖分连接相邻节点)
if num_graph_nodes_actual >= 3:
    try:
        # Delaunay 需要 (x, y) 坐标，我们使用 (lon, lat)
        delaunay = Delaunay(graph_node_coords)
        # Delaunay 三角形的边对应图中的边
        # 遍历所有三角形的边，添加为图的边
        added_edges = 0
        # delaunay.simplices 是一个数组，每一行是一个三角形的顶点索引
        for simplex in delaunay.simplices:
            # 三角形有三条边: (v0, v1), (v1, v2), (v2, v0)
            v0, v1, v2 = simplex[0], simplex[1], simplex[2]
            if not G.has_edge(v0, v1): G.add_edge(v0, v1); added_edges += 1
            if not G.has_edge(v1, v2): G.add_edge(v1, v2); added_edges += 1
            if not G.has_edge(v2, v0): G.add_edge(v2, v0); added_edges += 1

        print(f"通过 Delaunay 三角剖分添加了 {added_edges} 条边。")
        
        # 添加连接统计信息
        total_edges = G.number_of_edges()
        print(f"总边数: {total_edges}")
        print(f"图连通性: {'是' if nx.is_connected(G) else '否'}")
        print(f"平均度数: {2 * total_edges / num_graph_nodes_actual:.2f}")

    except Exception as e:
        print(f"Delaunay 三角剖分出错: {e}")
        traceback.print_exc()
        print("未能通过 Delaunay 添加边。图只包含节点。")
else:
     print("节点数量少于3个，无法进行 Delaunay 三角剖分添加边。图只包含节点。")


# 保存图
try:
    nx.write_graphml(G, GRAPH_OUTPUT_FILE)
    print(f"图已保存到 {GRAPH_OUTPUT_FILE}")
except Exception as e:
    print(f"保存图文件时出错: {e}")

graph_time = time.time() - graph_start_time
print(f"生成和保存图耗时: {graph_time:.2f} 秒。")


# --- 6. 生成可视化图 ---
print("\n开始生成可视化图...")
plot_start_time = time.time()
plt.figure(figsize=(10, 8))
ax = plt.gca()

# a. 绘制分析区域边界 (矩形)
# 由于我们现在是全国范围，绘制一个固定的边界框意义不大
# boundary_rect = patches.Rectangle((LON_MIN, LAT_MIN), LON_MAX - LON_MIN, LAT_MAX - LAT_MIN,
#                                  linewidth=1, edgecolor='black', facecolor='none', linestyle='-', label='分析区域边界')
# ax.add_patch(boundary_rect)

# b. 绘制过滤后的航迹点
# 使用原始过滤后的点 (不经过df_calc，包含所有点)
if not df_filtered.empty:
    ax.scatter(df_filtered['track_longitude'], df_filtered['track_latitude'],
               s=1, c='blue', alpha=0.3, label='Filtered Trajectory Points')
    print(f"绘制了 {len(df_filtered)} 个过滤后的航迹点。")

# c. 绘制图节点 (Original CD Points)，并用大小反映权重
if num_graph_nodes_actual > 0:
    node_lon = graph_node_coords[:, 0]
    node_lat = graph_node_coords[:, 1]
    # 绘制节点，使用归一化后的权重决定大小
    scatter = ax.scatter(node_lon, node_lat, s=normalized_weights, c='black', marker='o', label='Graph Nodes (by weight)')

    # 可选：添加节点 ID 或 权重标签
    # for i in range(num_graph_nodes_actual):
    #    ax.text(node_lon[i], node_lat[i], f"{i}", fontsize=8, ha='center', va='center') # 添加节点ID
    # for i in range(num_graph_nodes_actual):
    #     ax.text(node_lon[i], node_lat[i], f"{node_weights[i]:.1f}", fontsize=7, ha='center', va='bottom') # 添加权重值


    print(f"绘制了 {num_graph_nodes_actual} 个图节点，大小反映权重。")

    # d. 绘制Delaunay三角剖分的边
    if G.number_of_edges() > 0:
        print(f"绘制 {G.number_of_edges()} 条Delaunay边...")
        for u, v in G.edges():
            coord_u = graph_node_coords[u]
            coord_v = graph_node_coords[v]
            ax.plot([coord_u[0], coord_v[0]], [coord_u[1], coord_v[1]], 
                   'gray', alpha=0.5, linewidth=0.8)
        
        # 添加边的图例（只添加一次）
        ax.plot([], [], 'gray', alpha=0.5, linewidth=0.8, label='Delaunay Connections')

    # e. 绘制节点的凸包 (红色虚线边界)
    if num_graph_nodes_actual >= 3:
        try:
            hull = ConvexHull(graph_node_coords)
            # hull.vertices 按顺序给出凸包顶点的索引
            hull_points_indices = hull.vertices
            boundary_points = graph_node_coords[np.append(hull_points_indices, hull_points_indices[0]) , :] # 闭合环
            ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'r--', label='Convex Hull of Nodes')
            print("绘制了节点凸包边界。")
        except Exception as e:
            print(f"Error calculating or plotting convex hull: {e}") # QHull错误可能因为点共线等

    # 设置坐标轴、标题和图例
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Airspace Trajectories and Graph Node Extraction")
    # 适当设置坐标轴范围，可以稍微超出数据范围，或者使用自动调整
    # ax.set_xlim(LON_MIN, LON_MAX)
    # ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect('equal', adjustable='box') # 保持经纬度比例接近真实
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.legend(loc='upper right')
    plt.tight_layout()

    # 保存图像
    try:
        plt.savefig(PLOT_OUTPUT_FILE, dpi=300)
        print(f"可视化图已保存至 {PLOT_OUTPUT_FILE}")
    except Exception as e:
        print(f"保存可视化图时出错: {e}")

    plot_time = time.time() - plot_start_time
    print(f"生成和保存可视化图耗时: {plot_time:.2f} 秒。")
    # plt.show() # 如果需要交互式显示，取消此行注释

else:
    print("没有足够的图节点来生成可视化图。")


total_time = time.time() - start_process_time
print(f"\n总处理时间: {total_time:.2f} 秒。")