# 扩展二：让 nano-vllm 支持推测解码（Speculative Decoding）

本文讨论如何在当前 nano-vllm 的架构上支持推测解码。内容分四部分：

1. 推测解码的原理，以及它为什么能加速；
2. 它对引擎每一层提出了哪些新要求；
3. 参考项目 [nano-vLLM-MS2](https://github.com/banfeb/nano-vLLM-MS2)（下文简称 **MS2**）的实现；
4. 在当前项目中落地时的架构设计与优化方向。

阅读前建议先看 [basic/Scheduler.md](../basic/Scheduler.md)、[basic/block.md](../basic/block.md) 和 [basic/Sequence.md](../basic/Sequence.md)。推测解码几乎会碰到这几个组件的每一个设计点。

> 说明：MS2 基于较早版本的 nano-vllm（chunked prefill 重构之前），本文只借鉴它的设计思路，落地方案以当前项目的代码为准。对 MS2 的分析来自阅读源码，没有在 GPU 上实际运行。

---

## 一、原理

### 1. 为什么能加速

[basic/ContinuousBatching.md](../basic/ContinuousBatching.md) 中讲过：decode 阶段每步只生成 1 个 token，却要把整套权重从显存读一遍，瓶颈在**访存**而不在计算。

所以，**一次前向算 1 个 token 和算 4 个 token，耗时几乎一样**。推测解码利用了这一点：

```
普通 decode：  每步输入 1 个 token → 产出 1 个 token

推测解码：     ① 用一个很便宜的方法"猜"出后面 K 个 token（draft）
              ② 把"最后一个 token + K 个 draft"一起送进大模型，一次前向得到 K+1 个位置的 logits（verify）
              ③ 从前往后逐个检查 draft，接受连续的一段，在第一个被拒绝的位置补一个 token
              → 每步产出 1 ~ K+1 个 token，而前向的耗时和普通 decode 差不多
```

猜 draft 的方法有很多种：

| 方法 | 做法 | 代价 |
|---|---|---|
| **n-gram（prompt lookup）** | 在已有的上下文中找和"最后几个 token"相同的片段，把它后面的 token 当作 draft | 几乎为零，纯 CPU 字符串匹配 |
| 小模型（draft model） | 用同系列的小模型自回归生成 K 个 token | 需要第二个模型、第二份 KV cache |
| EAGLE / MTP | 在大模型的 hidden states 上接一个轻量的预测头 | 需要额外训练的权重 |

MS2 实现的是 **n-gram**。它对**输出和输入高度重合**的任务效果最好，比如代码修改、摘要、基于检索内容的问答。

### 2. 怎样保证输出分布不变：拒绝采样

推测解码的关键性质是：**最终输出的分布和只用大模型逐个采样完全相同**。做法如下（设第 i 个位置，大模型分布为 p，draft 分布为 q，draft token 为 d）：

```
以概率 min(1, p(d) / q(d)) 接受 d
若拒绝：从修正分布 r(x) ∝ max(0, p(x) − q(x)) 中采样一个 token 作为替代，本轮到此结束
若 K 个 draft 全部被接受：再从最后一个位置的 p 中额外采样一个 token（bonus token）
```

对 n-gram 来说，draft 是确定性的，相当于 q 是集中在 d 上的 one-hot 分布，上面的公式退化为：

- 接受概率就是 **p(d)**；
- 拒绝后从 **p 去掉 d 之后重新归一化**的分布中采样。

MS2 的 [docs/Speculative-Decoding.md](https://github.com/banfeb/nano-vLLM-MS2/blob/main/docs/Speculative-Decoding.md) 用"The capital of France is …"的例子，逐步演示了接受、拒绝和修正分布的计算过程，可以作为补充阅读。

### 3. 收益估算

设每个 draft 被接受的概率都是 α（简化为互相独立），每步提议 K 个 draft，则每步期望产出的 token 数为：

```
E[tokens/step] = 1 + α + α² + … + α^K = (1 − α^(K+1)) / (1 − α)
```

例如 α = 0.6、K = 3 时约为 2.2，也就是理想情况下 decode 步数减少到原来的 45% 左右。实际收益还要扣除两部分开销：

- **验证的开销**：batch 小时，前向的耗时由访存决定，多算几个 token 几乎不花时间；但 **batch 很大时，decode 也会逐渐变成计算瓶颈**，一次验证 K+1 个 token 的耗时会明显上升，收益随之缩小。
- **生成 draft 的开销**：n-gram 在 CPU 上运行，每步都会增加一些耗时。

所以推测解码在**小 batch、对延迟敏感**的场景中收益最大。MS2 报告在 Qwen3-0.6B 上吞吐从 1418 提升到 1997 tok/s（+40.8%）。这个数字和 prompt 内容、batch 大小、K 的取值都密切相关，不宜直接推广。

---

## 二、推测解码对引擎各层的要求

和 MoE 不同，推测解码**几乎触及每一层**。原因是它打破了 decode 阶段一个隐含的前提："每条序列每步只算 1 个 token，也只产出 1 个 token"。

| 层 | 原来的前提 | 推测解码之后 |
|---|---|---|
| **Sequence** | decode 时 `num_scheduled_tokens = 1` | 每步要计算 1 + k 个 token，但最终只确认其中一部分 |
| **BlockManager** | 每步最多新增 1 个块（`len % block_size == 1` 时） | 要一次为 1 + k 个 token 预留空间；被拒绝的 token 用过的块要**回滚** |
| **prefix cache** | 写满的块一定是有效内容 | 被拒绝的 token 的 KV 也写进了 cache，**绝不能登记 hash** |
| **Scheduler** | 每步 token 数 = 序列数 | 每步 token 数 = Σ(1 + k_i)，要计入 token 预算；后处理一次追加多个 token，EOS 可能出现在中间 |
| **ModelRunner** | decode 的 Q 长度为 1，可以用 CUDA Graph | Q 长度为 1 + k，需要新的输入准备逻辑，并且要考虑 CUDA Graph |
| **LMHead** | prefill 只取最后一个位置，decode 每条序列一行 | 验证时需要**每个位置**的 logits |
| **Sampler** | 每条序列采样一个 token | 拒绝采样，每条序列输出 1 ~ k+1 个 token |
| **TP 通信** | 每步传输 `seqs` | 还要把 draft 传给其他 rank |
| **新组件** | — | Proposer（生成 draft），只需在 rank 0 上运行 |

---

## 三、MS2 的实现

### 1. 一个推测解码 step 的完整流程

[llm_engine.py](https://github.com/banfeb/nano-vLLM-MS2/blob/main/nanovllm/engine/llm_engine.py) 的 `step()`：

```python
seqs, is_prefill = self.scheduler.schedule()
if not is_prefill and self.scheduler.is_spec_decoding:
    draft_token_ids = self.model_runner.propose_draft_token_ids(seqs)                  # ① 只在 rank 0 上运行
    draft_token_ids, reservations = self.scheduler.reserve_spec_decode(seqs, draft_token_ids)   # ② 预留 KV 块
    token_ids = self.model_runner.call("run_spec_decode", seqs, draft_token_ids, reservations)  # ③ 验证
    num_decode_tokens = self.scheduler.postprocess_spec_decode(seqs, token_ids, draft_token_ids, reservations)  # ④ 提交
else:
    ...                                                                                 # 原来的路径
```

prefill 完全不受影响。只有 decode step 改走新路径，原来的 `schedule()` 也基本不变。

### 2. 用一个例子跟踪

设 block_size = 4。某条序列共有 10 个 token `t0..t9`，其中 t0..t8 的 KV 已经写入 cache，t9 是上一步刚生成的。K = 3。

**① propose（n-gram）**：在 `t0..t9` 中找出与末尾最长匹配的片段，取它后面的 3 个 token 作为 draft `[d1, d2, d3]`。

MS2 的 [NgramProposer](https://github.com/banfeb/nano-vLLM-MS2/blob/main/nanovllm/v1/spec_decode/ngram_proposer.py) 把序列反转后用类似 KMP 的方式求最长匹配，长度限制在 `prompt_lookup_min` ~ `prompt_lookup_max` 之间，用 Numba 加速。总 token 数超过 8192 时启用多线程。

**② reserve**：`schedule()` 中的 `may_append` 已经为 t9 准备好了位置。现在还要为 d1..d3 预留位置：

```
需要写入 KV 的位置：9(t9) 10(d1) 11(d2) 12(d3)
现有 block_table 覆盖 0..11（3 块），位置 12 需要第 4 块 → 临时分配一个块，记在 reservation 中，暂不加入 block_table
如果空闲块不够：截短 draft，而不是抢占别的序列
```

**③ verify**：把这次验证**当作一次很小的 prefill** 来执行：

```
input_ids = [t9, d1, d2, d3]，positions = 9..12
cu_seqlens_q 每条序列加 4，cu_seqlens_k 每条序列加 13
slot_mapping 覆盖位置 9..12（包括临时预留的块）
→ 走 flash_attn_varlen_func + block_table，和 prefix cache 命中时的 prefill 完全相同
```

`LMHead` 新增了 `return_all_logits=True` 参数，取出这 4 个位置的 logits，按照 vLLM 的布局重新排列：**先放所有序列的 draft 行，再放所有序列的 bonus 行**：

```
t9 那一行 → 用来判断 d1      d1 那一行 → 判断 d2      d2 那一行 → 判断 d3      d3 那一行 → bonus token
```

然后交给 [RejectionSampler](https://github.com/banfeb/nano-vLLM-MS2/blob/main/nanovllm/v1/sample/rejection_sampler.py)，它由三个 Triton kernel 组成：生成均匀随机数、计算拒绝后的替代 token、逐位置判断接受或拒绝。

**④ commit**：假设 d1 被接受、d2 被拒绝，替代 token 为 r2：

```
输出 [d1, r2] → 依次 append_token，同时检查 EOS 和 max_tokens（停止条件可能出现在中间）
真正有效的 KV：t9 和 d1 → num_computed_tokens = 10 + 1 = 11
d2、d3 的 KV 也写进了位置 11、12，但已经无效：
  - 位置 11 下一步会被 r2 的 KV 覆盖
  - 第 4 块（只包含位置 12）不再需要 → 释放回空闲队列
只给 [0, num_computed_tokens) 范围内写满的块登记 hash → 被拒绝的内容永远不会进入 prefix cache
```

### 3. 关键设计点

| 设计点 | MS2 的做法 | 评价 |
|---|---|---|
| **验证走哪条执行路径** | 复用 prefill 的 varlen 路径 | 不需要新的 attention kernel，实现最简单；但不能使用 CUDA Graph（见第四节第 1 点） |
| **draft 存放在哪里** | 不写入 `Sequence.token_ids`，单独作为参数传给 ModelRunner | Sequence 只保存已确认的内容，回滚时不需要修改 token_ids |
| **KV 块的预留** | 先临时分配（reservation），验证后再决定保留多少、释放多少 | 预留和提交两步分离，回滚逻辑清晰 |
| **空间不够时** | 截短 draft，不抢占 | 推测解码只是一种优化，空间不够时退化成普通 decode，比为了它去抢占别的序列更合理 |
| **prefix cache 的安全性** | 只为有效范围内写满的块登记 hash | 正确。这是最容易出错的地方 |
| **Proposer 在哪里运行** | 只在 rank 0 上，以普通方法调用（不经过 `call` 广播） | 正确，其他 rank 只需要拿到 draft |
| **统计指标** | Scheduler 记录 `acceptance_rate`（不含 bonus token） | 便于调试 K 和 n-gram 参数 |

---

## 四、MS2 实现中值得改进的地方

### 1. 验证阶段不能使用 CUDA Graph

验证复用了 prefill 路径，而 prefill 在当前项目中一律走 eager 模式。MS2 的示例和 benchmark 也都设置了 `enforce_eager=True`。

这就产生了矛盾：推测解码要省的就是 decode 的步数，但验证步失去了 decode 最重要的优化手段（CUDA Graph），省下来的部分收益又被 kernel 启动开销吃掉了。

**改进：uniform decode 图**（vLLM 采用的做法）

- 每条序列的 Q 长度统一**补齐到 K+1**。draft 不足 K 个的序列用无效 token 填充，填充位置的 `slot_mapping` 设为 -1（[store_kvcache](../../nanovllm/layers/attention.py) 遇到 -1 会跳过写入），输出直接忽略；
- 这样输入形状就是规整的 `[bs, K+1]`。`flash_attn_with_kvcache` 本身支持 Q 长度大于 1（加上 causal mask），可以继续沿用 decode 的 attention 路径；
- 按 batch size 分档录制 `Q 长度 = K+1` 的图，和现有的 `capture_cudagraph` 机制完全一致。

代价是填充会浪费一些计算。但 decode 本来就是访存瓶颈，这点浪费通常可以忽略。

### 2. 序列化时总是传输完整的 token_ids

MS2 修改了 `Sequence.__getstate__`，**无论 prefill 还是 decode 都传输完整的 `token_ids`**。[basic/Sequence.md](../basic/Sequence.md#四__getstate__跨进程时只传必要的数据) 中分析过，当前项目在 decode 时只传 `last_token`，正是为了控制每步的 IPC 数据量。

验证时，worker 需要的输入只有"最后一个 token + draft"，而 draft 已经作为单独的参数传过去了，所以并不需要完整的历史。此外，共享内存只有 1MB，传输完整历史会让长序列更容易超出限制（见 [basic/ModelRunner.md](../basic/ModelRunner.md)）。

**改进**：保留当前项目的精简序列化方式，draft 单独传输。

### 3. 每步的 token 数没有计入预算

验证步一共要计算 Σ(1 + k_i) 个 token，但 MS2 的预留逻辑只检查 KV 块够不够，没有检查 token 预算。

[basic/Scheduler.md](../basic/Scheduler.md) 中讲过，`max_num_batched_tokens` 对应 warmup 测出的激活峰值。默认配置下 512 × (1+3) = 2048，远小于 16384，所以不会出问题。但如果 K 设得很大，或者调小了 `max_num_batched_tokens`，就可能超出预留的激活显存。

**改进**：在 Config 中加一条断言 `max_num_seqs × (1 + K) ≤ max_num_batched_tokens`，或者在调度时把 Σ(1+k_i) 计入 token 预算。

### 4. draft 的生成和 GPU 计算是串行的

每个 decode step 都要先在 CPU 上做完 n-gram 匹配，然后才开始 GPU 前向。batch 大、序列长时，匹配本身也要花时间。MS2 用 Numba 多线程缓解了这个问题，但没有消除串行等待。

**改进**（较复杂）：在 GPU 执行第 t 步的同时，CPU 为第 t+1 步准备 draft。这需要先知道第 t 步的结果，所以通常要配合"异步调度"一起设计，是 vLLM v1 中的做法。对 nano-vllm 来说可以作为后续优化。

### 5. draft 长度固定为 K

只有在 KV 空间不够时才会截短 draft。实际上，n-gram 对不同的序列、在不同的阶段，命中率差别很大。

**改进**：

- 根据每条序列最近的接受率，动态调整它的 K；
- 找不到匹配的序列（draft 为空）仍然按普通 decode 处理，Q 长度为 1。补齐成 uniform decode 时要注意，这种序列不应该拉高整体的填充开销。

### 6. 其他细节

- **Proposer 在所有 rank 上都被创建**：`NgramProposer` 的构造函数中会预分配 `(1024, max_model_len)` 的数组，并执行一次预热，但 rank > 0 上它从不会被调用。只在 rank 0 上创建即可。
- **不支持 greedy 解码**：当前项目的 `SamplingParams` 不允许 temperature = 0。如果将来支持 greedy，验证就退化为"argmax 是否等于 draft"，这时可以完全跳过随机数生成。

---

## 五、在当前项目中的落地设计

当前项目比 MS2 所基于的版本多了 chunked prefill 和 `hash_blocks`，Sequence 的进度字段也不同。下面给出按当前代码调整后的设计。

### 1. 核心思路：用现有的进度字段表达"推测"

当前项目已经有一套统一的进度表示（见 [basic/Sequence.md](../basic/Sequence.md#三其他场景下字段的变化)）：

- `num_cached_tokens`：KV 已经写入 cache 的 token 数；
- `num_scheduled_tokens`：本步要计算的 token 数；
- ModelRunner 按 `[num_cached_tokens, num_cached_tokens + num_scheduled_tokens)` 这个区间准备输入。

推测解码可以直接套用这套表示，只需要区分**"本步计算了多少"**和**"本步确认了多少"**：

```
schedule 时：   num_scheduled_tokens = 1 + len(draft)          ← 计算范围包含 draft
postprocess 时：实际确认的数量 = 1 + 被接受的 draft 数
                在调用 hash_blocks 之前，把 num_scheduled_tokens 改成实际确认的数量
                → hash_blocks 和 num_cached_tokens 的推进逻辑都不用改，只会处理已确认的部分
```

这样 `hash_blocks` 自然只会为已确认的内容登记 hash，第三节中提到的 prefix cache 安全问题也就不存在了。

### 2. 各组件的改动

| 组件 | 改动 |
|---|---|
| **Config** | 新增 `speculative_config`（method、K、n-gram 的最小/最大长度）；断言 `max_num_seqs × (1+K) ≤ max_num_batched_tokens` |
| **新增 Proposer** | `nanovllm/spec_decode/ngram_proposer.py`，接口为 `propose(seqs) -> list[list[int]]`。只在 rank 0 上创建，由 Scheduler 或 LLMEngine 调用 |
| **BlockManager** | 把 `can_append(seq)` / `may_append(seq)` 推广为 `can_append(seq, n)` / `may_append(seq, n)`：确保 block_table 能覆盖到位置 `num_cached_tokens + n - 1`。新增 `trim(seq)`：释放超出 `ceil(num_cached_tokens / block_size)` 的尾部块 |
| **Scheduler** | decode 分支中：先获取 draft，再按 `n = 1 + len(draft)` 检查空间；空间不够时**先截短 draft**，截到 0 仍然不够，才走原来的抢占逻辑。新增 `postprocess_spec`：逐个追加 token，处理中途的 EOS 或 max_tokens，修正 `num_scheduled_tokens` 后调用 `hash_blocks`，最后 `trim` |
| **Sequence** | 不需要新字段；draft 不写入 `token_ids`。`__getstate__` 保持现状 |
| **ModelRunner** | 新增 `prepare_verify`：输入为 `[last_token] + draft`，按 uniform decode 补齐到 K+1；新增验证用的 CUDA Graph 档位；`run` 需要额外接收 drafts 参数 |
| **Context / LMHead** | Context 中增加一个标志，告诉 LMHead 返回所有位置的 logits（MS2 用的是函数参数 `return_all_logits`，二者都可以） |
| **Sampler** | 新增 RejectionSampler（只在 rank 0 上运行）；n-gram 不需要 draft 的概率分布 |
| **LLMEngine** | `step()` 中根据是否开启推测解码选择路径；吞吐统计按实际产出的 token 数计算 |

### 3. 与 Scheduler 中其他机制的配合

| 机制 | 需要注意的地方 |
|---|---|
| **prefill 优先** | 推测解码只发生在 decode step，prefill 完全不变 |
| **chunked prefill** | 不受影响。最后一个 chunk 完成后，序列进入 running，下一步开始才可能推测 |
| **抢占** | 被抢占的序列 draft 自然作废。由于 draft 从未写入 `token_ids`，重新 prefill 时不会带入未确认的内容 |
| **prefix cache** | 依靠"先修正 `num_scheduled_tokens`，再调用 `hash_blocks`"来保证安全，见上文第 1 小节 |
| **max_tokens** | 一步最多可以产出 K+1 个 token，追加时要逐个检查，超过上限就截断 |
| **max_model_len** | draft 的位置不能超出 `max_model_len`。CUDA Graph 中 `block_tables` 的列数也是按它分配的，见 [basic/block.md](../basic/block.md#4-仍然可能出问题的边界情况) |

### 4. 向更强的 draft 方法扩展

如果之后要支持 draft model 或 EAGLE，架构上需要额外考虑：

- **Proposer 接口**要能返回 draft 的概率分布：`propose(seqs) -> (drafts, draft_probs | None)`。MS2 的 RejectionSampler 已经支持传入 `draft_probs`，可以直接复用；
- **draft model 自己的 KV cache**：最简单的做法是给它单独分配一份 KV 池，并且和主模型共用同一张 block_table（两个模型的 KV 写在各自 cache 中的相同位置），这样 BlockManager 不需要改动。代价是 KV 池要按两个模型的 KV 总和来规划容量；
- **draft model 自身也需要 decode**：K 次小模型前向同样需要 CUDA Graph，否则 kernel 启动开销会抵消大部分收益。

### 5. 推荐的实现顺序

1. Config + n-gram Proposer + Scheduler 的分支逻辑，先用"不做验证、总是拒绝"的模式跑通，确认输出和普通 decode **完全一致**；
2. 加入验证（先复用 prefill 路径，即 MS2 的做法）和 RejectionSampler，用 temperature 很低的设置对比普通 decode 的输出，检查接受率；
3. 推广 BlockManager 的 append/trim，用小 block_size 和小 KV 池专门测试回滚和抢占的边界情况；
4. 实现 uniform decode 的 CUDA Graph，对比开启前后的吞吐；
5. （可选）动态 K、异步生成 draft、draft model。
