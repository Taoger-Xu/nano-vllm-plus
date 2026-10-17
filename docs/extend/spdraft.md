# 推测解码设计 vs. 草稿小模型参考项目：改进点对照

本文把 [SpeculativeDecoding.md](SpeculativeDecoding.md) 中的设计与两个基于 nano-vllm 的草稿小模型参考项目逐条对比，用于实现时对照：

| 简称 | 仓库 | 特点 |
|---|---|---|
| **项目 A** | [limei1221/nano-vllm-speculative-decoding](https://github.com/limei1221/nano-vllm-speculative-decoding) | 较早的上游版本；verify 走 decode 路径并录 CUDA Graph；双 KV 池 |
| **项目 B** | [tianyuantong/nano-vllm-speculative-decoding](https://github.com/tianyuantong/nano-vllm-speculative-decoding) | 与本项目基于同一上游版本（`bb823b3`）；批量 GPU 采样、每轮一次 D2H；verify 走 varlen 路径并录 CUDA Graph；测试和基准做得很完整 |

> 前提：设计文档中的设计还没有完整实现，下面的"改进"都是设计层面的分析，没有实测数据支撑。对两个参考项目的分析来自阅读源码和它们的文档，没有在 GPU 上实际运行。

> **更新（2026-10-17）**：草稿模型的 KV 管理已从"两个模型共用一张 block_table"改为"两套 KVState、两个 BlockManager"，与项目 B 的思路相同，并在开启草稿模型时关闭 prefix cache（见 [temp.md](temp.md) 第一节、设计文档五.4、五.5）。第一、二部分中"设计文档"一列关于共用 block_table、Q=2 补齐和 B-1 的描述，记录的是当时的方案，保留作对比；以第三部分和设计文档为准。

---

# 第一部分：与项目 A 的对比

## 一、直接沿用的部分

有几处项目 A 做得很好，设计文档直接采用了：

| 内容 | 设计文档中的位置 |
|---|---|
| 验证走 decode 路径：把 q reshape 成 `(B, K+1, H, D)`，交给 `flash_attn_with_kvcache`，并单独录制 verify 的 CUDA Graph | 第五节第 9 小节 (1)(2)(4) |
| 调度时按 `len + K` 预留空间，被拒绝后多出来的块不释放，留给下一步继续用 | 第五节第 7 小节 |
| 拒绝采样的向量化写法：`ratio` → `cumprod` → 修正分布或 bonus token | 第五节第 3 小节 |

## 二、改进点

| # | 方面 | 项目 A | 设计文档 | 好处 | 设计文档中的位置 |
|---|---|---|---|---|---|
| 1 | **草稿模型的 KV 块管理** | 两套块：`draft_blocks`、`free_draft_block_ids`、`draft_block_table`，与主池同步分配 | 两个模型**共用一张 block_table**，同一块号在两个池中各自寻址 | 按它的显存划分公式，两个池的块数本来就相等，两套管理维护的是同一份信息。共用之后 BlockManager 不需要知道草稿模型的存在，而且**草稿模型自动获得 prefix cache**（项目 A 的草稿池不做 prefix cache，命中前缀时仍要重算整个 prompt） | 五.4 |
| 2 | **草稿模型如何追上进度** | 用 `draft_kv_len` 记录落后多少；只要 batch 中有一条序列落后，第 0 步就**整批**走 eager 的 varlen prefill | prefill 时两个模型一起算；decode 时第 0 步**固定输入最后 2 个 token**（Q=2） | 不需要额外字段，第 0 步的形状永远是 `(B, 2)`，可以录 CUDA Graph。项目 A 中新序列加入、或某条序列 K 个全部被接受时，都会触发 eager 路径 | 五.4、五.5 |
| 3 | **谁修改 Sequence** | ModelRunner 把草稿 `append_token` 进序列，验证后再 `pop`；Scheduler 后处理时再截断一次；`draft_kv_len` 两边都要维护 | 草稿不写入 `token_ids`；ModelRunner 只返回 token 列表；**只有 Scheduler 的 `postprocess_spec` 修改状态** | 回滚逻辑集中在一处，`token_ids` 始终只包含已确认的内容，抢占后重新 prefill 不会带入未确认的 token。这也符合 nano-vllm 原有的分工 | 五.6、五.9 (5) |
| 4 | **prefix cache 的 hash 登记** | `_ensure_blocks` 只在追加新块时检查**当前最后一块**是否写满。由于预留了 K 个位置，有的块写满时已经不是最后一块，**永远不会被登记** | 先把 `num_scheduled_tokens` 改成确认的数量，再调用现有的 `hash_blocks`，按范围登记 | 不会漏登记，命中率不受影响；`hash_blocks` 本身不用改。两种做法都不会把被拒绝的 KV 登记进去，所以这一条只影响命中率，不影响正确性 | 五.6 |
| 5 | **张量并行** | 只有 rank 0 进入推测路径，其他 rank 走普通 decode，前向的 token 数对不上，all-reduce 会挂住，**TP > 1 不可用** | `run_spec` 通过 `call` 在所有 rank 上执行，草稿 token 用 `dist.broadcast` 同步 | 架构上支持 TP。不过设计的第一版同样断言 TP = 1，这一条目前只是留好了扩展的位置 | 五.10 |
| 6 | **大 batch 时自动关闭** | 总是推测。它自己的测试显示 B ≥ 64 时只有约 0.59× | running 超过 `max_batch_size` 时，本步走普通 decode | 避免大 batch 时变慢；关闭时可以直接用原来的 decode 图 | 五.8 |
| 7 | **接近 `max_model_len` 的序列** | 没有检查（据读到的代码），位置 `len + K` 可能超出 rotary 表和图中 `block_tables` 的列数 | 任何序列超出时，整批退回普通 decode | 堵住一个越界的边界情况 | 五.8 |
| 8 | **CUDA Graph 的内存池** | 大模型 decode、草稿 decode、verify 三组图**各用一个池** | 所有图**共用一个池**，先录最大的 verify 图 | 图的总显存约等于最大的那个图，而不是三组之和 | 五.9 (4) |
| 9 | **warmup 与显存规划** | warmup 只跑大模型的 prefill，草稿模型的激活不计入峰值 | prefill 时两个模型一起算，warmup 自然把草稿模型算进去 | KV cache 预留得更准确 | 五.4 |
| 10 | **配置与校验** | 平铺的 `speculative_model`、`num_speculative_tokens`；**不校验词表是否一致** | `SpeculativeConfig`，按方法分别校验；断言词表一致、`max_num_seqs × (1+K) ≤ max_num_batched_tokens` | 词表不一致时启动即报错，而不是静默出错；验证步的激活不会超过 warmup 测出的峰值 | 三.2 |
| 11 | **统一框架** | 只支持草稿模型 | n-gram 和草稿模型共用调度、验证、拒绝采样和后处理，通过 `method` 切换 | 增加一种草稿来源（例如以后的 EAGLE / MTP），只需要实现 propose | 四、五.1 ~ 五.3 |
| 12 | **attention 的改动方式** | 增加 `num_spec_tokens > 0` 的特殊分支 | `Context` 增加 `q_len`，decode 分支统一为 `q.view(-1, q_len, H, D)` | 一处改动覆盖普通 decode、验证、草稿第 0 步和后续步四种情况 | 五.9 (2) |
| 13 | **统计指标** | decode 吞吐仍按 `len(seqs)` 计算，推测时显示偏低；没有接受率 | 按实际产出的 token 数计算吞吐，并统计接受率 | 调整 K 和 `max_batch_size` 时有依据 | 五.11 |

## 三、这些改进的代价

改进并不都是免费的：

- **Q=2 的第 0 步**：每一步都多算 1 个 token。对小模型来说影响不大，但确实比项目 A"只在落后时追赶"多做了计算；
- **prefill 时两个模型一起算**：每次 prefill 都要多跑一遍草稿模型。项目 A 推迟到第一次推测时才追赶，如果请求很短、几乎没有 decode，项目 A 的做法反而更省；
- **整批统一决定是否推测**：比逐条序列决定要粗糙。只要一条序列接近 `max_model_len`，整批都会退回普通 decode。好处是 verify 的形状统一，补齐的位置一定落在已分配的块里；
- **共用 block_table**：有一个小的例外。如果序列恰好在"K 个全部接受"的那一步结束，最后一个位置的草稿 KV 永远不会补上，之后命中这个块的请求会读到旧数据。这只会降低接受率，输出依然无损（设计文档第五节第 5 小节）；
- **按 batch 大小关闭推测、保留 prefix cache**：这两个功能带来一个新问题。不推测的 decode 步如果只运行大模型，草稿模型的 KV 就会出现大段空洞，Q=2 只能补 1 个 token（见第八节 B-1）。

## 四、总结

项目 A 解决了"让小模型打草稿跑起来并且快"这个问题，尤其是 verify 能录 CUDA Graph，这一点设计文档直接沿用了。设计文档的改进主要集中在三个方面：

- **和 nano-vllm 现有架构的契合**：共用 block_table，沿用进度字段和 `hash_blocks`，由 Scheduler 统一修改状态（改进 1、3、4）；
- **边界情况**：TP、大 batch、`max_model_len`、词表校验（改进 5、6、7、10）；
- **CUDA Graph 的覆盖率和显存**：Q=2 的第 0 步、共享内存池（改进 2、8、12）。

---

# 第二部分：与项目 B 的对比

项目 B 有两个版本。第一版（多个 PR 拼出来的离线控制器）只做到与普通 decode 持平；作者分析瓶颈后重写了一版，接入上游的调度器，README 报告的结果如下（Qwen3-8B + Qwen3-0.6B，BF16，RTX PRO 6000 Blackwell，Qwen3 推荐的采样参数，k = 3）：

| max_num_seqs B | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 相对普通 decode 的吞吐 | 1.50× | 1.37× | 1.24× | 1.17× | 1.04× |

## 五、整体路线

| | 项目 B | 设计文档 |
|---|---|---|
| 草稿来源 | **只支持草稿模型**；n-gram 在重写时被删掉了 | n-gram 和草稿模型统一在一个框架中 |
| 采样 | temperature、**top-k、top-p**、**greedy** | 只有 temperature（当前的 `SamplingParams` 不允许 greedy） |
| 代码组织 | 两个独立的 `ModelRunner`（target 和 draft，通过 `kv_role` 区分）+ 单独的 `SpeculativeDecoder` 类 | 一个 `ModelRunner` 同时持有两个模型，新增 `run_spec` |
| 配置 | `draft_model`、`num_speculative_tokens` 等平铺字段；**必须显式指定两个模型各自的 KV 显存预算** | `SpeculativeConfig`；按两个模型每块的字节数之和自动划分 |

## 六、关键设计点的差异

| # | 设计点 | 项目 B | 设计文档 | 评价 | 设计文档中的位置 |
|---|---|---|---|---|---|
| 1 | **KV 块管理** | 两套 `KVState` 和两个 `BlockManager`，各自分配 | 两个模型共用一张 block_table | 设计文档更简单。项目 B 的两个池大小独立，所以要求用户手动给两个显存预算 | 五.4 |
| 2 | **prefix cache** | 开启草稿模型时**必须关闭** | 两个模型都支持，命中的块在两个池中都有效 | 设计文档覆盖了更多场景，但也因此要处理草稿 KV 空洞的问题（第八节 B-1） | 五.4 |
| 3 | **草稿模型追赶进度** | 记录 `draft_kv.num_cached_tokens`，满足"落后 0 或 1 个"的不变量；对落后的序列**单独跑一次草稿 decode 图**来补上 | 第 0 步固定 Q=2，把补齐合并进第一次草稿前向 | 思路相同：都利用了"最多落后 1 个"。项目 B 多一次前向（约 2ms），但不需要额外的图；设计文档少一次前向，但多录一组 Q=2 的图。项目 B 第一版用 eager 追赶，每轮约 10ms，印证了避免 eager 追赶的必要性 | 五.5 |
| 4 | **验证走哪条路径** | **prefill 的 varlen 路径**（`flash_attn_varlen_func` + block_table），**同样录成了 CUDA Graph** | decode 的 `flash_attn_with_kvcache`，q reshape 为 `(B, K+1, H, D)` | 见下方说明 | 五.9 (1)(2) |
| 5 | **补齐的行** | 额外分配一个**全零的 KV 块** `pad_block_id`，补齐行的 block_table 指向它 | `slot_mapping = -1`；`context_lens = len + K` | 项目 B 让补齐行读到确定的有限值。设计文档的补齐行输出虽然会被丢弃，但同样应确认不会产生 NaN 或越界 | 五.9 (1) |
| 6 | **按 batch 大小关闭** | 没有。只要配置了草稿模型，每一步 decode 都推测 | running 超过 `max_batch_size` 时走普通 decode | 项目 B 的数据（B=16 时约 1.0×，k=4 时 0.989×）印证了这个开关的价值 | 五.8 |
| 7 | **接近 `max_model_len`** | `add_request` 时直接拒绝 `len(prompt) + max_tokens + k > max_model_len` 的请求 | 整批退回普通 decode | 项目 B 的做法更简单，彻底消除了这个边界情况，值得借鉴 | 五.8 |
| 8 | **EOS / max_tokens** | 每条序列总是提议 k 个，多出来的在提交时丢弃 | 相同 | 一致 | 五.6 |
| 9 | **采样** | 整批在 GPU 上向量化处理，**每轮只做一次 D2H**（一个 `[B, k+2]` 张量）；去掉了 `torch.compile` | 向量化的 `RejectionSampler`，最后一次 `.tolist()` | 一致。项目 B 的性能分析说明了这一点有多重要：第一版逐请求在 Python 中采样，每个请求约 2.8ms，B=4 时这部分开销就相当于一次 8B 模型的前向 | 五.3 |
| 10 | **谁修改序列状态** | `postprocess_speculative` 统一提交，但 `_catch_up` 会在 `SpeculativeDecoder` 中直接修改 `draft_kv.num_cached_tokens` | 只有 Scheduler 修改 | 设计文档更严格 | 五.6 |
| 11 | **测试和基准** | CPU 单元测试（采样频率与精确分数参考值对比）；GPU 上三道检查：验证 logits 与逐步 decode 的数值噪声、采样分布、greedy 等价性；完整的 benchmark 协议，包括 TTFT、TPOT 和每阶段耗时 | 只有实现顺序中的几个验证点 | 项目 B 扎实得多，值得照搬 | 七.3 |

### 关于第 4 点：varlen 路径也能录图

[cuda-graph.md](../basic/cuda-graph.md) 中写过"prefill 的 varlen 路径中 `max_seqlen` 是 Python 标量，所以不能录图"。项目 B 给出了一个例外：验证时每条序列的 Q 长度固定为 k+1，于是它把 `max_seqlen_q` 固定为 k+1、`max_seqlen_k` 固定为 `max_model_len`（实际长度由每行的 `cu_seqlens_k` 限定），`cu_seqlens` 也放进固定的缓冲区。标量参数不再变化，varlen 路径也就能录图了。

两种做法的取舍：

- **项目 B**：attention 完全不用改；
- **设计文档**：要在 `Context` 中加 `q_len`，但 decode 和验证共用同一个 kernel，草稿模型的 Q=2 和 Q=1 也能统一处理。

把 `max_seqlen_k` 固定为上限是否会影响 kernel 的调度和性能，项目 B 的文档中没有分析。

### 为什么项目 B 没有草稿 KV 空洞的问题

项目 B **没有按 batch 大小关闭推测**，也**禁用了 prefix cache**：每一步 decode 都会运行草稿模型，也不存在命中草稿 KV 不完整的块的情况。设计文档同时保留了这两个功能，所以必须补上"不推测的步也让草稿模型写 KV"（第八节 B-1）。

## 七、项目 B 的性能分析：一轮的耗时都花在哪里

项目 B 的 [PERFORMANCE.md](https://github.com/tianyuantong/nano-vllm-speculative-decoding/blob/main/docs/PERFORMANCE.md) 把一轮推测的耗时分解为"固定开销 + 每请求开销 × B"。第一版在 B=4 时一轮约 30ms：

| 组成 | 约多少 | 说明 |
|---|---|---|
| 大模型的验证前向（CUDA Graph） | 11ms | 8B BF16 权重 16.4GB，在 1.79 TB/s 带宽下理论下限约 9.2ms，已达到约 83%，基本无法再压缩 |
| 3 次草稿前向 + 追赶（0.6B，CUDA Graph） | 5ms | `compute_logits` 和逐步采样在图外 |
| 主机同步、锁页内存拷贝等 | 4ms | 重写后改为每轮一次 D2H |
| 每请求的 Python 采样 | 约 2.8ms × B | 重写后改为整批向量化，基本消除 |

对设计文档的启示：推测解码能否赚到，取决于"每轮产出的 token 数 ÷ 一轮的耗时"能否超过普通 decode。验证前向的开销是固定的，可以优化的是草稿前向、同步次数和采样这几部分。这也是设计文档坚持"草稿模型要录图、采样要向量化、只同步一次"的原因。

---

# 第三部分：实现对照清单

## 八、从两次对比中得到的待办事项

| # | 事项 | 来源 | 影响设计文档的位置 |
|---|---|---|---|
| B-1 | ~~不推测的 decode 步也运行一次草稿模型~~ **不再需要**：改为两套独立管理后，草稿模型的进度显式记录在 `draft_kv` 中，不推测时允许落后，重新推测时一次补齐 | 代码检查 + 项目 B 的对比 | 五.5 |
| B-2 | `add_request` 时检查 `len(prompt) + max_tokens + K ≤ max_model_len`，代替"整批退回普通 decode" | 项目 B | 五.8 |
| B-3 | 补齐行指向一个全零的 KV 块，或至少确认补齐行不会产生 NaN 或越界 | 项目 B | 五.9 (1) |
| B-4 | 验证路径的备选方案：固定 `max_seqlen_*` 后用 varlen 路径录图；同时更正 cuda-graph.md 中"varlen 路径不能录图"的说法 | 项目 B | 五.9、cuda-graph.md |
| B-5 | 三道 GPU 检查：验证 logits 与逐步 decode 的数值噪声基线、采样分布的频率测试、greedy 等价性 | 项目 B | 七.3 |
| B-6 | 按阶段计时（catch_up / propose / verify / accept / copy），用"固定开销 + 每请求开销 × B"分解一轮的耗时 | 项目 B | 五.11 |
| B-7 | （可选）支持 top-k / top-p；草稿模型和大模型使用相同的截断 | 项目 B | 五.3 |

## 九、实现清单

按设计文档第七节第 3 小节的实现顺序排列，每完成一项打勾。括号中是对应的改进编号：数字是第一部分的改进点，B-x 是第八节的待办事项。已勾选的项是对照代码检查过的。

**第 1 步：配置与骨架**
- [x] `SpeculativeConfig`：按方法校验、词表一致、token 预算断言（10）
- [ ] `add_request` 时检查长度上限（B-2）
- [ ] `Context.q_len` + attention decode 分支按 `q_len` reshape（12）
- [x] `BlockManager.can_append(seq, n)` / `may_append(seq, n)`：预留 `len + K`，多余的块不释放（沿用）
- [x] Scheduler decode 分支：按 batch 大小、`max_model_len` 整批决定是否推测（6、7）。目前由 `spec_enabled = False` 关闭
- [ ] `postprocess_spec`：草稿不进 `token_ids`，先修正 `num_scheduled_tokens` 再 `hash_blocks`（3、4）
- [ ] 验证：`num_drafts = 0` 时输出分布与关闭推测解码时一致

**第 2 步：n-gram + 拒绝采样**
- [x] `NgramProposer`（11）
- [ ] `RejectionSampler`：向量化写法，`q` 为 `None` 时按 one-hot 处理，每轮一次 D2H（沿用、11）
- [ ] verify 走 decode 路径（沿用；备选方案见 B-4），补齐行的处理（B-3）
- [ ] 验证：重复性强的 prompt 接受率明显大于 0

**第 3 步：草稿模型**
- [x] 两套 `KVState`（`Sequence.target_kv` / `draft_kv`）、两个 BlockManager 同进同退；开启草稿模型时关闭 prefix cache；`allocate_kv_cache` 按两个模型的块字节数之和分配，两个池块数相同
- [x] prefill 时草稿模型按自己的块表单独准备输入并前向，warmup 计入草稿模型的激活（9）。注意：warmup 目前跳过了 `compute_logits` 和 `Sampler`
- [ ] 草稿模型按 `draft_kv.num_cached_tokens` 追赶：第 0 次补齐落后部分（常见情况 Q=2），之后 Q=1。`DraftModelProposer` 目前没有处理落后
- [x] ~~不推测的 decode 步也运行草稿模型（B-1）~~ 不再需要
- [ ] 验证：草稿模型 = 大模型时，接受率接近 100%

**第 4 步：边界测试**
- [ ] 抢占（包括只有草稿池不够）、prefix cache 命中（n-gram）、`max_tokens` / EOS 出现在草稿中间、接近 `max_model_len`、从不推测切回推测时的大段追赶
- [ ] 三道 GPU 检查（B-5）

**第 5 步：CUDA Graph**
- [ ] verify 图 + 两组草稿图，所有图共用一个内存池，先录最大的 verify 图（8）

**第 6 步：统计与调参**
- [ ] 按实际 token 数统计吞吐，打印接受率（13）
- [ ] 按阶段计时（B-6）
- [ ] 在 B = 1、8、16、32、64 上实测，确定 `max_batch_size` 和 K 的默认值（6）

**第 7 步（可选）**
- [ ] TP > 1：`run_spec` 在所有 rank 上执行，草稿 token `dist.broadcast`（5）
- [ ] top-k / top-p（B-7）
