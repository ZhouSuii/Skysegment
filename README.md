1. environment.py: 你的 GraphPartitionEnvironment 类。如果算法 B 需要状态是 PyG 的 Data 对象，可能需要修改这个文件。
2. agent_dqn_basic.py: 实现算法 A 的 DQN 智能体。
3. agent_gnn.py: 实现算法 B 的 GNN-based 智能体 (例如 PPO+GNN)。
4. metrics.py: 编写计算你选择的评估指标的函数 (方差、切割边、模块度等)。这些函数应该接收图对象 graph 和分区分配数组 partition_assignment 作为输入。
5. baselines.py: 编写运行随机划分和 METIS/KaHIP 的函数。
6. run_experiments.py: 主要的脚本，负责加载图、运行各种算法、收集结果。
7. configs/ (可选): 存放配置文件 (数据集路径、算法超参数等)。
8. results/: 存放实验结果数据 (CSV 文件)。
9. plots/: 存放生成的图表 (PNG 文件)。



### SOME EXPERIENCE
1. 超参数问题：
    - episode: 训练的总回合数
        - 增加：提高解决方案质量，但延长训练时间
        - 减少：加快训练，但可能导致结果不佳
    - max_steps: 每个回合的最大步数
        - 增加：允许更多探索，但可能导致过拟合
        - 减少：加快训练，但可能导致结果不佳
    - batch_size: 每次训练的样本数量
        - 增加：提高训练稳定性，但增加内存消耗
        - 减少：加快训练，但可能导致结果不佳
    - gamma: 折扣因子
        - 增加：更关注长期奖励，但可能导致短期奖励被忽视
        - 减少：更关注短期奖励，但可能导致长期奖励被忽视
    - learning_rate: 学习率
        - 增加：加快收敛速度，但可能导致不稳定
        - 减少：提高稳定性，但收敛速度变慢
    - epsilon: 探索率
        - 增加：更多探索，但可能导致不稳定
        - 减少：更关注利用，但可能导致局部最优
    - target_update_interval: 目标网络更新频率
        - 增加：提高稳定性，但收敛速度变慢
        - 减少：加快收敛速度，但可能导致不稳定
2. batch_size与gpu利用率的关系：
在gnn模型中，调大batch_size会导致gpu利用率下降的问题很常见，问题分析：  
图批处理较为特殊，首先不规则的图结构会导致内存访问模式不连续，增大batch可能会加剧问题；
其次，每次_state_to_pyg_data调用都会创建新的numpy数组，执行循环操作，cpu->gpu传输数据，batch变大会增加串行操作的占比
3. 在gnn训练过程中发现传统架构下的dqn有210.85it/s但是在gnn模型下只有50it/s的速度，原因如下：  
    - 计算复杂度高了很多：
        - GNN 需要进行图卷积操作，涉及邻居节点信息的聚合
        - 每一层 GCNConv 的计算复杂度与边数成正比
        - 传统 DQN 只进行简单的矩阵乘法
    - 数据结构的差异：
        - GNN 使用的 PyTorch Geometric 数据结构更复杂
        - 为每个样本构建 PyG Data 对象开销很大
    - 内存分配和数据传输:
        - 图批处理（Batch）操作比简单向量批处理复杂得多
        - 在 GPU 和 CPU 之间传输图结构数据更耗时
    - 批处理机制：
        - GNN 的批处理需要处理不同大小的图和节点索引
        - Batch.from_data_list() 操作需要处理每个图的边索引的偏移量
4. 训练过程中发现gpu利用率还是很低(5-10%)，但是cpu利用率非常高(105%)  
说明计算瓶颈在cpu上而非gpu(4090)，可能原因如下：
    - 数据预处理和环境交互 (CPU密集型):
        - 环境模拟：GraphPartitionEnvironment 中的 step、reset 和 _calculate_reward 方法涉及图的操作（如检查边、计算权重、更新分配），这些都是在CPU上执行的，大型图会消耗大量cpu时间
        - 状态转换：将环境状态 (partition_assignment, 节点权重/度) 转换为模型所需的输入格式（扁平化的Numpy数组或PyTorch Geometric的Data对象）是在CPU上完成的。特别是 _state_to_pyg_data 函数（在 agent_gnn.py 和 agent_ppo_gnn.py 中），它需要为每个状态构建节点特征矩阵并创建Data对象，这可能非常耗时。虽然 agent_ppo_gnn.py 中进行了一些优化（如预计算固定特征），但这仍然是一个主要的CPU任务。
    - 数据传输瓶颈：
        - 模型（如DQN, GNNDQN, PPOPolicy, GNNPPOPolicy）虽然在GPU上运行，但每次调用 act（推理）或 replay/update（训练）时，都需要将当前状态或一批经验数据从CPU内存传输到GPU内存。
        - 对于GNN模型，需要传输Data对象或Batch对象。频繁的小批量数据传输会产生显著的开销，CPU忙于准备和传输数据，而GPU则处于等待状态。
    - 模型计算量较小:
        - 对于某些图的大小和复杂度，你使用的模型（即使是GNN）在GPU上的前向和后向传播可能非常快。强大的GPU（如RTX 4090）可以在极短的时间内完成计算
        - 如果CPU准备下一批数据的时间远长于GPU处理当前批次的时间，GPU就会大部分时间处于空闲状态
    - Agent 内部的 CPU 计算:
    - Python GIL 和串行操作:

5. skysegment与test对比：
    - dqn_basic中episode=10000，sky只需要不到一分钟，但是epsd=500 test要五分钟
    - 

6. 性能分析：
   - 使用tensorboard和Nsight systems进行性能分析
   - Nsight System：nsys profile --stats=true -o /home/zhousui/pyproj/ python run_experiments.py
     - CPU 时间主要分散在 Python 解释器本身、CUDA/PyTorch 的 CPU/CUDA 后端库以及标准 C 库中。这再次印证了 CPU 在执行高层逻辑、进行系统调用和与 GPU 交互时花费了大量时间。高 CPU 利用率 (99.99% 的进程利用率) 并不意味着 CPU 在有效计算
       结合 osrtsum 的报告，它更可能意味着 CPU 线程在忙碌地等待（spin-waiting 或处于等待状态）。
     - CUDA Driver/Nsight Systems 版本不匹配 (Warning 4): 这是之前无法看到 GPU Kernel 和 Memory Activity 数据的主要原因
     - 无法跟踪 Unified Memory (Error 5): 可能与版本不匹配或硬件/配置有关。解决版本不匹配问题可能也会解决或改变这个错误
7. 尝试思路：
    - 超参数调优AHPO：使用optuna来进行四个算法的超参数化调优
    - 尝试将runexperiments拆分成四个脚本，一次运行多个算法看看能不能提高gpu占用和减少运行时间
        - 运行方式 CUDA_VISIBLE_DEVICES=0/1 python run_shell.py [num_nodes]/[json_location]
        - 失败，虽然在一个gpu上同时执行两个任务会让gpu占用大幅上升，但是流多处理器利用率提高并未带来速度的提升，思路失败

8. 经验之谈：
    - 安装Nsight Systems
        - 直接wget最新版，要注意版本兼容
        - 注意内核要配置perf_event_paranoid选项，否则不支持某些CPU性能追踪功能
    


算法的实现与优化都感觉一个非常非常庞大的任务，很多时候都是在摸黑操作，像计算机一样的精密结构牵一发而动全身