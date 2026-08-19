# ANN-RL 算法框架（当前实现）

> 这份文档描述**代码里现在真正在跑的东西**。`framework.md` 是最初的设计草稿，保留作为出处，
> 但它已经和实现有实质差异（候选池流水线、接受测试、长度头都是后加的）。
>
> **维护约定**：改动 `lib/algorithm.py` / `lib/agent.py` / `lib/reward.py` 的语义，或新增 CLI 开关时，
> 同步更新对应模块的 **I/O 契约**和 §10 参数速查表。实验结论写在 [EXPERIMENTS.md](EXPERIMENTS.md)，
> 这里只写**机制**——但每个组件标注它的证据状态，避免把已证伪的部件当成有效设计。
>
> 最后更新：2026-08-10（加入 D2 长度头；澄清逐节点 rank 的实现代价；每个模块补 I/O 契约）

---

## 0. 记号

| 符号 | 含义 | 典型值 |
|---|---|---|
| `N` | 顶点数 | 100,000 |
| `deg` | `graph.max_degree`，出度上限 | 24 |
| `B` | 本步采样的节点数 `nodes_per_step` | 512 |
| `C` | 2-hop 候选池宽度（每步由批内最大池决定） | ~460 |
| `M` | `n_rand_cand`，追加的均匀候选数 | 256 |
| `emb` | 顶点嵌入维度 | — |
| `nq` | `probe_size`，探针查询数 | — |
| `n_swap` | 每个节点换几条边 | 2 |
| `pad` | 邻接表填充值 | −1 |

**约定**：`_np` 后缀 = numpy（CPU），无后缀 = torch tensor（`self.device`）。
`cand_ids` 的**填充槽也持有合法顶点索引**（不是 −1），因为下游无论 mask 与否都会嵌入每一列。

---

## 1. 一句话

把「构建 ANN 近邻图」变成一个 MDP：状态是整张图，动作是**每个节点换掉几条出边**，
奖励是换边前后**同一批探针查询**的检索质量差，用 PPO 学这个换边规则。

目标命题：从任意 $s_0$ 出发迭代收敛到的图，优于人工构图基线。
（现状见 EXPERIMENTS.md §1——命题 (1) 成立，命题 (2) 尚未成立。）

---

## 2. MDP 定义

| 元素 | 定义 | 形状 |
|---|---|---|
| **状态 $s_t$** | 有向图邻接表 `hnsw.adj` | `[N, deg]` int，pad=−1 |
| **动作 $a_t$** | 每个采样节点：①是否行动 ②新边长度分位 ③加哪 `n_swap` 条 ④删哪 `n_swap` 条 | 见 §4.6 |
| **转移** | $s_{t+1} = \text{accepted} \;?\; \text{edited} : s_t$ | — |
| **奖励 $r_t$** | $c(s_t) - c(s_{t+1})$，再按"探针走过哪些节点"分配到节点 | `[B]` float32 + `[B]` bool 观测掩码 |

**出度恒定**：换边是"删 k 条、加 k 条"，槽位被覆写而非删除。所以任何比较都自动是度数匹配的。

---

## 3. 一个训练步

代码入口 `GraphEditPPO.train_step(probe_queries, probe_gt)`（`lib/algorithm.py`）。

```
                probe_queries [nq, d], probe_gt [nq, k]
                              │
① 采样节点      ──────────────┼──►  node_ids_np [B] int64
② 编码全图      state = agent.prepare_state(graph)  →  state.vertices  z [N, emb]
③ 构造动作      sample_swaps(state, node_ids_np)    →  action dict  (§4)
④ c(s_t)        probe_costs(...)  →  pre_rewards [nq]，有缓存
⑤ 转移          hnsw.apply_swaps(...)               →  num_changed int
⑥ c(s_{t+1})    probe_costs(...)  →  post_rewards [nq]
                delta = post_rewards − pre_rewards     [nq] float
⑦ 信用分配      credit_nodes(...) →  rewards [B] float32, observed [B] bool
⑧ 接受/回滚     resolve_acceptance(...) → accept_frac float
⑨ PPO 更新      →  mean_reward float | None（无观测节点时返回 None）
```

**探针**：从 SIFT 的 learn set 抽 `probe_size` 条查询，每 `probe_resample_every` 步重抽。
④ 的结果被缓存（图没动时不重算），所以每步只有**一次**额外搜索开销。

**commit_stride**：>1 时同一个 $s_t$ 被采样 K 次、都用于训练，但只有最后一次真正留下。
它的回滚**不看奖励**，和接受测试是两回事。

---

## 4. 动作构造：`sample_swaps` ← **本项目最关键的部分**

```
输入   state        agent.prepare_state 的返回值（.vertices = z [N, emb]）
       node_ids_np  [B] int64
       greedy       bool，True 时全部取 argmax 不采样
输出   action dict（下表），或 None（批内没有一个节点有可用候选）
```

这条流水线决定了策略**能选什么**。五轮实验的结论是：瓶颈从来在这里，不在打分器
（EXPERIMENTS.md §6）。

### 4.1 `hnsw.get_candidates` — 2-hop 邻域

```
输入   node_ids   [B] int64
输出   cand_ids   [B, C] int64   ← 填充槽持有索引 0（合法），不是 −1
       cand_mask  [B, C] bool
```
排除节点自身和它的 1-hop 邻居（那些已经是边）。
⚠️ kNN 图上**每节点最长候选只到长度分位 8.95**——这是 D1/D2 必须追加随机候选的原因。

### 4.2 `augment_candidates` — 加宽 + 按长度带筛选

```
输入   node_ids_np  [B] int64
       cand_ids_np  [B, C] int64
       cand_mask_np [B, C] bool
       p_center     [B] float | None   ← None 走全局 CDF 口径，给值走逐节点 rank
输出   cand_ids_np  [B, C+M] int64     ← 变宽
       cand_mask_np [B, C+M] bool
副作用 self._cand_rng_seed += 1（保证跨步不同行）
```

依次做四件事：
1. 追加 `M = n_rand_cand` 个均匀采样顶点
2. 剔除自环和已有边（保证"加边"总是真的新边）
3. 剔除外围离群点（`max_periph_pct`）
4. 按长度分位保留

### 4.3 两种分位口径**不等价**

| 口径 | 定义 | 触发 |
|---|---|---|
| **全局 CDF** | 边长 d 的分位 = 20 万随机点对里有多少比它短。**所有节点同一把尺** | `cand_band (LO,HI)`，D1 |
| **逐节点 rank** | d 在"从 u 出发的距离分布"里的分位。**自动吸收局部密度差异** | `p_center` 给值，C2/D2 |

C2 优化的是逐节点 rank，D1 实现的是全局 CDF，两者只在数据完全均匀时重合。

> **⚠️ 逐节点 rank 的实现代价——不要按字面理解成"给全部 N 个点排序"。**
>
> 概念上是"把其余 N−1 个点按到 u 的距离排序，取第 p% 位"，那样是 O(N)/每条边、O(N²) 总量，
> 大数据集上完全不可行。**训练循环里不这么做**：
> ```python
> samp = np.sort(distances(node, row[C:]))          # M 个均匀样本
> pct  = 100.0 * np.searchsorted(samp, d) / len(samp)
> ```
> 均匀样本的经验分位是总体 rank 的**无偏估计**，成本 **O(M)，与 N 无关**。
> 分位估计误差约 `sqrt(p(1−p)/M)`：M=256 → ~3.0 个分位点，M=1024 → ~1.5；策略用的带宽 ±10。
> 同一批点既当候选池又当参照集，自含偏差 ~1/M ≈ 0.4 个分位点，可忽略。
>
> **已实测**（`c2_sampled.py`）：采样版 M=64 → 0.5483、M=256 → 0.5482，全量 argsort → 0.5520 ± 0.0103。
> **规则本身是 O(M) 的，不是 oracle。**
>
> 测量脚本（`c2_lengthsweep.py` / `stage0_adaptive_p.py` / `stage1_eval.py`）用全量 `cdist().argsort()`
> 只是因为一次排序能服务所有 p 值和所有 draw，是测量便利、不是算法要求。

**兜底规则**：带内候选不足 `n_swap` 时，取分位**最接近带中心**的若干个，而不是退回未过滤的池
——退回会把短边重新放上菜单，恰好是给最需要带过滤的节点。全行被 mask 会让 `log_softmax` 出 NaN。

### 4.4 `filter_long_candidates` — B2 的长边过滤（可关）

```
输入   node_ids_np  [B], cand_ids_np [B, C'], cand_mask_np [B, C']
输出   cand_mask_np [B, C']    ← 只改 mask，cand_ids 不动
```
只保留每行最长的 `long_frac` 比例。`long_frac=0` 时直接原样返回。

### 4.5 `enough` 检查

```
条件   cand_mask.sum(-1) >= max(n_swap,1)   且   当前出度 > n_swap
效果   不满足的节点整行丢弃 → B 变小为 B'
```

### 4.6 打分与采样

```
node_ctx(state, node_ids, adj_ids, adj_mask)      → ctx [B', emb] | None
score_candidates(state, node_ids, cand_ids,
                 cand_mask, ctx, in_deg)          → logits [B', C'] float32，mask 处 −inf
```

| 决策 | 采样方式 | 输出 |
|---|---|---|
| 长度 `p_u` | $\mathcal{N}(p_\text{mean}, \sigma)$ | `p_sample [B]` float |
| 加边 | Gumbel-top-k over `logits` | `chosen [B', n_swap]` 列索引 |
| 删边 | Gumbel-top-k over `−nb_logits` | `drop_slots [B', n_swap]` 邻接槽位索引 |
| 门 | `rand < sigmoid(act_logits)` | `act [B']` bool |

**长度头在最前面跑**，因为它的输出定义了菜单——`p_sample` 要先算出来才能传给 `augment_candidates`。

### 4.7 action dict 的键

| 键 | 形状 | 用途 |
|---|---|---|
| `node_ids` / `node_ids_np` | `[B']` | 本步实际参与的节点 |
| `cand_ids` / `cand_mask` | `[B', C']` | 更新时重打分要用**同一批**候选 |
| `chosen` | `[B', n_swap]` | 加边的列索引 |
| `adj_ids` / `adj_mask` | `[B', deg]` | **$s_t$ 的快照**，更新时重算 ctx 和删边 logits |
| `drop_slots` | `[B', n_swap]` | 删边的槽位索引 |
| `in_deg_np` | `[N]` | $s_t$ 的入度快照 |
| `act` / `act_np` | `[B']` bool | 门决策 |
| `p_sample` | `[B']` float | D2 采样出的分位，更新时重算 log-prob |
| `swap_node_ids_np` | `[B'']` | 只含 `act=True` 的节点，交给环境 |
| `new_nb_ids` / `drop_nb_ids` | `[B'', n_swap]` | 交给 `apply_swaps` 的顶点 id |
| `old_logp` | `[B']` | 采样时的联合 log-prob，PPO ratio 的分母 |
| `next_adj_np` | `[B', deg]` | 仅 critic 用，$s_{t+1}$ 的邻接 |

⚠️ `adj_ids` 必须是快照：更新在几个 epoch 之后跑，那时图已经是 $s_{t+1}$ 了。

---

## 5. 智能体（`MLPLinkAgent`，`lib/agent.py`）

```
prepare_state(graph, device, training)  →  state，其中 state.vertices = z [N, emb]
```
编码器**只看坐标，不看邻接**。这就是为什么 critic 必须吃邻域均值——只吃 `z_i` 的话
$V(i,s_t) \equiv V(i,s_{t+1})$，TD 目标会退化。

| 头 | 输入 | 输出 | 初始化 |
|---|---|---|---|
| **打分器** | `z_u [B,emb]`, `z_v [B,emb]` | `logit_scale·dot(f(z_u),f(z_v))` → `[B]` | 预训练 |
| `get_node_ctx` | `node_ids [B]`, `adj_ids [B,deg]`, `adj_mask [B,deg]` | `ctx [B, emb]` | — |
| `ctx_head` | `[z_u, z_v, ctx_u, |z_v−ctx_u|]` | 对打分的残差修正 `[B]` | **零初始化** |
| `get_act_logits` | `node_ids [B]`, `adj_ids`, `adj_mask`, `in_deg [N]` | `[B]`（sigmoid 前） | 零权重，bias=+2 |
| `get_values` | `node_ids [B]`, `adj_ids [B,deg]`, `adj_mask [B,deg]` | `V [B]` | **零初始化** |
| `get_len_delta` | `node_ids [B]`, `adj_ids [B,deg]`, `adj_mask [B,deg]` | `δ [B]` → `p_u = p0+span·tanh(δ)` | **零初始化** |

**零初始化残差是这个项目里唯一反复有效的模式**：开关打开时策略**精确等于**打开前的策略，
所以任何差异都是严格改进测试，不是从零赌一把。`actor_ctx`（长期唯一有效的干预）和
`len_head`（第二个正结果）都用了它。

**打分器的架构性限制**：`dot(f(z_u), f(z_v))` 中 f 平滑 ⇒ 打分必然随距离单调，
ρ(score, len) 在**随机初始化**下是 −0.80，比预训练后的 −0.58 **更**偏短。
所以**任何只改权重的方案都逃不掉短边偏好**——这条否掉了换 loss / 换监督 / 换 encoder 的整条思路。

`len_head` **不受**这条限制：逐节点标量，没有 pairwise 结构、不排候选。

---

## 6. 奖励与信用分配

### 6.1 搜索

```
hnsw.search_deterministic(queries [nq, d])
  → best_vertex_ids              [nq, k]      返回的近邻
    total_distance_computations  [nq]
    num_hops                     [nq]
    trajectories                 [nq, max_trajectory]   走过的顶点序列
```

### 6.2 图级性能（`ProximityDCSReward`）

```
输入   res（上面的 dict）, ground_truth [nq, k]
输出   rewards [nq] float  +  terms（各分项，用于诊断）
```

$$R = R_{\text{target}} + \alpha R_{\text{dense}} + \beta R_{\text{cost}}$$

- $R_{\text{target}}$ = recall@k，硬正确性信号
- $R_{\text{dense}}$（`dense='prox'`）= $\text{mean}_j\, d(q, gt_j) / d(q, found_j) \in [0,1]$，
  **答案的近似比**。recall 为 0 时仍能区分"差一点"和"离谱"，且**只是答案的函数**，不能靠多走路刷分
- $R_{\text{cost}} = -(\text{DCS}/\text{DCS}_{\max})$，`--beta 0` 时关闭（预算已由 `dcs_budget` 硬约束）

⚠️ `dense='path'`（Spearman 路径单调性）是**遗留项**，实测与 recall **反向**（NSW 上 −0.354），
只保留用于复现旧实验。

### 6.3 `credit_nodes`

```
输入   node_ids_np [B'] int64
       delta       [nq] float      = R(s_{t+1}) − R(s_t) 逐查询
       pre_res, post_res           两次搜索的 dict（要它们的 trajectories/num_hops）
       act         [B'] bool
输出   rewards     [B'] float32
       observed    [B'] bool
```

把 $\Delta_q$ 分摊给"换边前**或**后的搜索路径**走过**的节点"（按访问次数平均）。

> **没被任何探针走到的节点是"未观测"，整行从更新中 mask 掉，不是记 0。**
> 记 0 会让绝大多数节点提供一个假的"这个动作无害"信号。

---

## 7. 接受测试

```
resolve_acceptance(action, rewards, observed, delta, undo, ...)  →  accept_frac float
副作用：按模式回滚部分或全部节点的邻接行
```

`--accept {off, step, node}` — 这是**改变 MDP 本身**的开关，不只是记账。
没有它，转移是**无条件**的：从一个已经是局部最优的图出发，唯一可能的轨迹就是下坡
（NSW 上实测：300 步 ≈ 13.4 万条平均有害的编辑，recall −0.045）。

| 模式 | 语义 |
|---|---|
| `node` | 只保留自身 credited reward 为正的节点的编辑 |
| `step` | 按聚合 delta 全有或全无 |
| `off` | 无条件提交 |

被拒绝的提案**仍然是训练信号**（策略学的是提案的实测 delta），只是不再伤害图。

**训练用 `node`，部署用 `off`**：同一个冻结策略在 `off` 下收益是 `node` 的 5.6 倍。

**回滚机制**：`hnsw.snapshot(node_ids [B'']) → dict`，`hnsw.restore(dict) → None`。
只快照会被写的行，不是整张邻接表。

---

## 8. 动作的因子分解

四个决策独立，所以联合 log-prob 是精确的求和：

$$\log \pi(a_t) = \underbrace{\log P(\text{gate})}_{\text{sigmoid}} + \underbrace{\log P(p_u)}_{\text{Gaussian}} + \mathbb{1}[\text{acted}]\big(\underbrace{\log P(\text{adds})}_{\text{PL}} + \underbrace{\log P(\text{drops})}_{\text{PL}}\big)$$

```
plackett_luce_logp(logits [B,C], chosen [B,n_swap])  →  [B]
gaussian_logp(x, mean, sigma)                        →  与输入同形
masked_logp(act [B], logp [B])                       →  [B]，act=False 处置 0
```

- **加边**：Gumbel-top-k 采样 ≡ Plackett-Luce（无放回有序抽取），对 logits 可导
- **删边**：同一个打分器，`softmax(−score)` 上 Gumbel-top-k。
  一个边质量函数同时服务两侧：高分 = 值得留 = 容易被加、不容易被删
- **长度**：**不被 gate 门控**——p 在 gate 之前采样并塑造了菜单，门控它会让采样端和更新端的因子分解不一致
- **无操作节点**的 log-prob 照样进 ratio：不编辑也是一个真实决策

⚠️ `drop_mode` 为 `argmin`/`random` 时删边项是 θ 的常数，在 ratio 里约掉，**必须排除在 old_logp 外**。

**采样端和更新端的因子分解必须逐字一致**，否则 $\exp(\log\pi - \log\pi_{\text{old}})$
比较的是两个不同的分布。更新端重算时用**快照的 $s_t$ 邻接行**，不是当前图。

---

## 9. PPO 更新

```
输入   action dict, rewards [B'] , observed [B']
输出   mean_reward float | None
```

标准 clipped surrogate，`ppo_epochs` 轮，每轮重编码一次、mini-batch 累积梯度。

- **优势**：`reward − baseline`。baseline 是逐节点 EMA（`SessionBaseline`），或 `gamma>0` 时用 critic 的 TD(0)
- ⚠️ **`adv_ref='noop'` 时不做均值中心化**。中心化会强制 mean(advantage)=0，
  于是无论所有动作是不是都有害，总有一半被强化。无操作的奖励**恒等于 0**，
  它是让绝对符号有意义的固定参照
- 熵奖励：加边侧和删边侧分开计（`entropy_reg` / `drop_entropy_reg`），因为删边侧的熵奖励把分布推向均匀，
  而均匀删边正是 `drop_mode='policy'` 要打败的基线
- 优化器健康探针：梯度范数、参数位移、GradScaler 跳步数、`advantage_std_raw`
  （归一化前的信号强度，接近 1e-8 时梯度实际已死）

---

## 10. 参数速查（附证据状态）

✅ 有效 ｜ ⚪ 无效/零结果 ｜ ❌ 有害 ｜ 🔬 在验

### 动作空间（**证据显示这里是主要杠杆**）
| 开关 | 作用 | 状态 |
|---|---|---|
| `--n_rand_cand N` | 每行追加 N 个均匀采样候选 | ✅ 没有它长度带不可达（带内 0%） |
| `--cand_band LO HI` | 全局 CDF 长度带 | ✅ D1，0.5403 |
| `--len_head` | 学逐节点长度分位 | 🔬 TEST 2 通过（+0.032, t=3.5） |
| `--len_p0/span/sigma/half` | 带中心 / 幅度 / 探索 / 半宽 | 🔬 span 是关键，见 EXPERIMENTS.md |
| `--len_fixed P` | 同一条 rank 带但固定 P（**len_head 的匹配对照**） | — |
| `--max_periph_pct P` | 剔除质心距离分位 > P 的候选 | 🔬 均值未分辨，但跨 seed sd 砍 4.6× |
| `--long_frac F` | 只留 2-hop 池中最长的 F | ✅ 0.299→0.502，但打不过匹配随机 |

### 智能体
| 开关 | 状态 |
|---|---|
| `--actor_ctx` | ✅ +0.0081（4 seed 配对）——长期唯一有效的干预 |
| `--logit_scale S` | ✅ 必需。`dot` 返回 [−1,1] 余弦，240 候选的 softmax 在 scale=1 时最高只有均匀的 1.2 倍 |
| `--indeg_ctx / --indeg_noop` | ⚪ **三次测成 null**，最后一次还是逐点精确匹配入度 |
| `--scratch` | 诊断用：随机初始化反而**更**偏短边（−0.80 vs −0.58） |

### 训练
| 开关 | 状态 |
|---|---|
| `--accept node` | ✅ 训练必需（+0.0529 ± 0.0070，7.6σ）；**部署要关** |
| `--dense prox` | ✅ 默认；`path` ❌ 与 recall 反向 |
| `--drop_mode` | ⚪ `policy` 与 `argmin` 不可区分 |
| `--gamma` / critic | ⚪ 三个折扣值都没救回 NSW；critic 能工作，但折扣从来不是瓶颈 |
| `--seed` | ⚠️ A/B 必须设。不设的话两臂起点就差 0.0418 平均奖励，比要测的效应还大 |

---

## 11. 不变量与坑

1. **邻接行必须尾部紧凑**。`search_hnsw.cc` 读到第一个 `-1` 就停，中间的 pad 会静默截断邻居表。
2. **`cand_ids` 的填充槽必须是合法索引**（不是 −1），下游无条件嵌入每一列。
3. **候选行不能全被 mask**，否则 `log_softmax` 出 NaN 并污染熵项。
4. **`reachable` 是无界 BFS**，饱和在 ~0.994，无信息。真正排序 recall 的是**预算内 coverage**（rho=+1.0）。
5. **训练日志里的 recall 不可跨 run 比较**（各自 batch）。一切结论过 `eval_harness.py`。
6. **边长一律用分位表述**。注意：**均匀随机目标在分位 47，不是 100**。
7. **免学习对照是强制的**。当前 bar：kNN + 5% 边重连到长度分位 65 = **0.5520 ± 0.0103**。
   对照必须**按编辑量**匹配，并跨 ≥3 个 slot seed 平均。

---

## 12. 文件地图

| 文件 | 内容 |
|---|---|
| `lib/algorithm.py` | `GraphEditPPO`：MDP、候选流水线、采样、信用分配、接受测试、PPO |
| `lib/agent.py` | `MLPLinkAgent`：编码器与五个头 |
| `lib/reward.py` | `ProximityDCSReward` |
| `lib/hnsw.py` | `GraphEditHNSW`：邻接、换边、快照/回滚、确定性搜索 |
| `lib/search_hnsw_swig/` | C++ 搜索内核（SWIG 绑定） |
| `train_sift100k_ppo.py` | CLI 与主循环 |
| `eval_harness.py` | **唯一评分入口**（多 seed 对照 + coverage + add_pct） |
| `lib_control.py` | 免学习对照 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 实验复盘：每个组件为什么在这里、哪些被证伪 |
