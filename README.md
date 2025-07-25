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
     - 在超参数调优时我们在代码中添加了时间计算，发现在每个trial中前向传播的实现都接近90%-
     其中前向传播时间主要消耗在gcn以及后续的actor与critic线性层


7. 尝试思路：
    - 超参数调优AHPO：使用optuna来进行四个算法的超参数化调优
    - 尝试将runexperiments拆分成四个脚本，一次运行多个算法看看能不能提高gpu占用和减少运行时间
        - 运行方式 CUDA_VISIBLE_DEVICES=0/1 python run_shell.py [num_nodes]/[json_location]
            - 失败，虽然在一个gpu上同时执行两个任务会让gpu占用大幅上升，但是流多处理器利用率提高并未带来速度的提升，思路失败

8. 经验之谈：
    - 安装Nsight Systems
        - 直接wget最新版，要注意版本兼容
        - 注意内核要配置perf_event_paranoid选项，否则不支持某些CPU性能追踪功能
    
9. 后续优化：
    - 视觉效果优化：
        - 图表：解决matplotlib中文显示问题或者改为英文显示
        - 图表：使用更高分辨率的图像 + 增大字体
        - 航路轨迹显示：优化中文显示
    - 代码优化：
        - 使用pbrs的思路引入势函数来修改奖励计算方式
        - 对奖励的超参数进行调优
        - 对算法的超参数进行调优
    - 问题处理：
        - gpu占用不高/cpu占用高 -- 模型训练速度慢
        - 引入gnn改进后算法效果负优化

10. 待完成需求：
    - 25/4/30: 对五个tpo代码进行理解与注释，重新考虑数据的生成与处理方式  -- finished
    - 25/5/1: 重构四个agent的hpo代码支持新的reward函数      -- finished
    - 25/5/1: 在gnn中实现usecudastreams 分析并优化前向传播占比时间过长的问题  -- 改进前向传播一次搜集数据
    - 25/5/11: 修正gnnppo训练时loss为0的问题，删除update_frequency并引入n_steps来进行更新  -- finished
    - 25/5/11: 发现pbrs_edge_cut_weight 和 pbrs_modularity_weight 的重要性极低，可能需要修改reward -- 修改了tpo的objective value
    - 25/5/16: 修改gnnppo: 包括减少gnn网络层数，在gnn层添加残差连接，实现ppo值函数裁剪，可能需要添加聚合特征与修改网络参数初始化
    - 25/5/17: 分析与稳定gnn模块，现在在加入gnn模块后ppo与dqn都从收敛变到无法收敛：
        - step1：检查gnn的输出是否合理  --- gcn层与actor/critic层输出的嵌入范数持续增大
        - step2：梯度流检查  --- 梯度爆炸与梯度范数剧烈变化/消失
        - step3：gnn架构再评估
    - 25/5/28: 
        - 动作空间与状态表示不匹配：GNN学习的是"节点i应该分配到分区j"的概率，但动作空间把它当作20个独立的动作选择。GNN输出是 [num_nodes, num_partitions] 的概率矩阵，但环境期望的是单一的整数动作，语义层面的不匹配。
        - PBRS势函数设计优化：分区权重方差、模块度、切割边三个指标上要表现好，但它们往往是冲突的
        - 价值函数聚合问题：
    
    -25/6/5:
        - 重新设计gnnppo算法，将gnn作为一个特征提取器，注意：
            - 状态表示的改变
            - 修改ppo网络结构与前向方法
            - 动作空间与Actor网络的输出
    -25/6/7: 
        - 设计自适应的GNN架构 adaptive.py
        - 全面测试框架 comprehensive_test.py
        - 诊断工具 quick_diagnosis.py
        - 注意设置随机种子，确保实验可重复性
    -25/6/22：
        - 将gnnppo的学习率下降一个数量级，算法开始收敛且收敛速度远快于ppo -- 可能是ppo优化不够
        - 将updatefrequecy从9提升到2048，速度非常大提升，批处理思想
    -25/7/3:
        - partitions=2 使用原始数据图
        - 使用大型数据集，交代elbow method
        - 实现viewpartitions，分析结果 --- achieve
    -25/7/4:
        - 明确我们使用强化学习的目的：
            - 并不是要在指标上完全超越metis
            - metis是一个启发式的算法，其划分规律是确定的
            - 我们使用强化学习，是为了论证方法的有效性
            - 强化学习的思路更加灵活，调整策略就可以适配不同的划分需求


算法的实现与优化都感觉一个非常非常庞大的任务，很多时候都是在摸黑操作，像计算机一样的精密结构牵一发而动全身