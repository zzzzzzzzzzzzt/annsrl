## 1. 节点特征融合 (Feature Fusion)

首先，模型需要将节点的**空间几何特征**与**图拓扑特征**进行融合：
- **输入特征**：
    - 原始空间坐标特征：$x_i \in \mathbb{R}^D$
    - 经由多层 NodeFormerConv 提取的图拓扑特征：$U_i \in \mathbb{R}^d$
        - 代码中原来在 NodeFormer 层间传递的隐藏变量 `z`，在该设计里应理解为拓扑表征 $U_i$，而不是最终融合后的节点表征。
- **特征对齐与融合**：
    1. **维度对齐**：在所有 NodeFormerConv 层之后，利用一个多层感知机 MLP（由一个线性层 Linear 和一个 Sigmoid 激活层组成），将拓扑特征 $U_i$ 的维度从 $d$ 映射到 $D$，得到对齐后的特征 $\text{MLP}(U_i) \in \mathbb{R}^D$。
    2. **特征融合**：将原始坐标 $x_i$ 与对齐后的拓扑特征进行**逐元素相乘（Element-wise Product）**，实现特征的自适应加权，最后与 $x_i$ 进行**残差相加**，得到最终的节点表征 $z_i \in \mathbb{R}^D$。
    $$z_i = x_i \odot \text{MLP}(U_i) + x_i$$

## 2. 边预测推理 (Edge Prediction Inference)

基于融合后的最终节点表征 $z_i$，模型利用**线性注意力机制**（Kernel-based Linear Attention）来高效计算节点 $i$ 和节点 $j$ 之间存在边的概率 $\pi_{ij}$：
$$\begin{aligned}
\pi_{ij}
&= \frac{\exp\left(\left((z_i)^T W_1 W_2 z_j + g_j\right) / \tau\right)}{\sum_k^N \exp\left(\left((z_i)^T W_1 W_2 z_k + g_k\right) / \tau\right)} \\
&= \frac{\Phi\left((z_i)^T W_1 / \sqrt{\tau}\right) \Phi\left(W_2 z_j / \sqrt{\tau}\right) \exp(g_j / \tau)}{\Phi\left((z_i)^T W_1 / \sqrt{\tau}\right) \sum_k^N \Phi\left(W_2 z_k / \sqrt{\tau}\right) \exp(g_k / \tau)}
\end{aligned}$$
- **参数说明**：
    - $W_1, W_2$：可学习的权重参数矩阵，用于将融合后的 $z_i \in \mathbb{R}^D$ 投影到边预测空间。
    - $g$：偏置项（或先验地理/拓扑偏置）。初始实现可以先不显式引入额外偏置，只保留稳定的 Gumbel 扰动。
    - $\tau$：控制概率分布集中度的温度超参数（Temperature）。
    - $\Phi(\cdot)$：核函数（Kernel Function），用于将 Softmax 的指数项拆解为线性内积，从而将计算复杂度从 $\mathcal{O}(N^2)$ 降至 $\mathcal{O}(N)$。

## 3. 对比学习损失函数设计 (Contrastive Loss)

模型通过对比学习（Contrastive Learning）来监督边的预测，通过拉近正样本对、拉远负样本对来优化网络：
- **样本采样策略**（针对中心节点 $i$）：
    - **正样本（$P_i$）**：节点 $i$ 的真实邻居节点（1-hop 邻居）。
    - **负样本（$N_i$）**：只采用节点 $i$ 的 2-hop 非邻居节点，即可以通过两跳到达、但不是 $i$ 的直接 1-hop 邻居、也不是 $i$ 自身的节点。
- **目标损失函数**：
    为了避免单一正样本主导损失，我们将多正样本的对比损失形式修正为**对数平均交叉熵**。即对每一个真实邻居（正样本 $j^+$）分别计算其在“正样本 + 2-hop 负样本”候选空间中的相对概率，再在正样本集 $P_i$ 上求平均：

$$Loss_i = -\frac{1}{|P_i|} \sum_{j^+ \in P_i} \log \frac{\pi_{ij^+}}{\sum_{j^- \in N_i} \pi_{ij^-} + \pi_{ij^+}}$$

实现时，若某个节点没有可用的 2-hop 非邻居负样本，应跳过该中心节点，或将该节点的损失置零并从最终平均分母中排除。
