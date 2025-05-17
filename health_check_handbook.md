# GNN-PPO 健康检查指南

本指南介绍如何使用新添加的 GNN-PPO 健康检查功能来诊断模型性能问题。

## 功能概述

GNN-PPO 健康检查系统包含以下主要功能：

1. **嵌入统计监控**：收集和展示 GNN 层输出的均值、标准差和范数
2. **梯度流监控**：检测梯度是否存在消失或爆炸问题
3. **节点嵌入可视化**：使用 PCA 降维，生成节点嵌入的 2D 可视化
4. **快速训练模式**：减少 episode 数量，更快获取诊断结果

## 如何运行健康检查

有两种运行健康检查的方式：

### 1. 使用专用脚本

```bash
# 基本用法
./run_health_check.py

# 自定义参数
./run_health_check.py --episodes 100 --max-steps 50 --nodes 30 --partitions 3
python run_health_check.py --episodes 50 --max-steps 50 --nodes 10 > ./logs/check_$(date +%Y%m%d_%H%M%S).log 2>&1
```

参数说明：
- `--episodes`：要运行的 episode 数量（默认：50）
- `--max-steps`：每个 episode 的最大步数（默认：100）
- `--nodes`：图中的节点数量（默认：20）
- `--partitions`：分区数量（默认：2）

### 2. 通过原始实验脚本

```bash
python run_experiments.py --mode health-check --episodes 50 --max-steps 100 --nodes 20 --partitions 2
```

## 健康检查输出

运行健康检查会生成以下输出：

1. **控制台输出**：
   - GNN 各层的嵌入统计信息
   - 梯度范数
   - 训练指标（奖励、损失、方差）
   - 警告信息（如梯度爆炸或消失）

2. **可视化文件**：
   - `results/embeddings/layer1_epX.png`：第一层 GNN 嵌入的 2D 可视化
   - `results/embeddings/layer2_epX.png`：第二层 GNN 嵌入的 2D 可视化
   - `results/plots/gnn_ppo_health_check.png`：训练曲线图

## 如何解读结果

### 1. 嵌入统计

```
[Episode 5] GNN Embedding Health Check:
Layer 1 GCN: Mean=0.1234, Std=0.4567, Norm=2.3456
Layer 2 GCN: Mean=0.2345, Std=0.5678, Norm=3.4567
Actor Output: Mean=0.3456, Std=0.6789, Norm=4.5678
Critic Output: Mean=0.4567, Std=0.7890, Norm=5.6789
```

- **正常范围**：
  - 均值应在 -1.0 到 1.0 之间
  - 标准差应在 0.1 到 1.5 之间
  - 范数应随层数增加而适度增长，但不应突然爆炸

- **问题迹象**：
  - 均值远离 0（>1.0 或 <-1.0）
  - 标准差过小（<0.01）或过大（>10）
  - 范数突然爆炸增长（比如从 2 到 1000）

### 2. 梯度统计

```
Gradient Norms:
conv1_grad_norm: 0.2345
conv2_grad_norm: 0.3456
actor_grad_norm: 0.4567
critic_grad_norm: 0.5678
```

- **正常范围**：大多数参数的梯度范数应在 0.001 到 10 之间
- **问题迹象**：
  - 梯度消失：范数过小（<0.0001）
  - 梯度爆炸：范数过大（>100）

### 3. 嵌入可视化

观察 `results/embeddings` 目录下的可视化图像：

- **良好的嵌入**：节点形成清晰的结构，相似节点聚集在一起
- **问题嵌入**：
  - 所有节点挤在一起（过度平滑）
  - 节点完全随机分布（特征不相关）
  - 极端离群值（数值不稳定）

## 诊断和修复建议

如果发现问题，以下是一些可能的解决方案：

1. **过度平滑问题**：
   - 减少 GNN 层数（已实施）
   - 增加或调整残差连接（已实施）
   - 更换激活函数（尝试 ELU 或 LeakyReLU）

2. **梯度问题**：
   - 降低学习率
   - 添加梯度裁剪（已实施，但可能需要调整阈值）
   - 使用不同的优化器（如 RAdam、AdamW）

3. **数值不稳定**：
   - 在前向传播中添加批量归一化
   - 检查状态表示的缩放
   - 确保 PPO 裁剪参数合理

4. **训练不稳定**：
   - 增加 entropy 系数促进探索
   - 调整奖励缩放
   - 尝试基于势的奖励整形（PBRS）

## 参考资料

- [PyTorch Geometric 文档](https://pytorch-geometric.readthedocs.io/)
- [PPO 算法论文](https://arxiv.org/abs/1707.06347)
- [GNN 过度平滑问题论文](https://arxiv.org/abs/2006.11468)
