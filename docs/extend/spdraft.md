# 推测解码设计 vs. 草稿小模型参考项目：改进点对照

本文对比 [SpeculativeDecoding.md](SpeculativeDecoding.md) 中的设计与草稿小模型参考项目 [limei1221/nano-vllm-speculative-decoding](https://github.com/limei1221/nano-vllm-speculative-decoding)（下称"原项目"），用于实现时逐条对照。

> 前提：文档中的设计还没有实现，也没有跑过，下面的"改进"都是设计层面的分析，没有实测数据支撑。对原项目的分析来自阅读源码。

---

## 一、直接沿用的部分

有几处原项目做得很好，设计文档直接采用了：

| 内容 | 设计文档中的位置 |
|---|---|
| 验证走 decode 路径：把 q reshape 成 `(B, K+1, H, D)`，交给 `flash_attn_with_kvcache`，并单独录制 verify 的 CUDA Graph | 第五节第 9 小节 (1)(2)(4) |
| 调度时按 `len + K` 预留空间，被拒绝后多出来的块不释放，留给下一步继续用 | 第五节第 7 小节 |
| 拒绝采样的向量化写法：`ratio` → `cumprod` → 修正分布或 bonus token | 第五节第 3 小节 |

## 二、改进点

| # | 方面 | 原项目 | 设计文档 | 好处 | 设计文档中的位置 |
|---|---|---|---|---|---|
| 1 | **草稿模型的 KV 块管理** | 两套块：`draft_blocks`、`free_draft_block_ids`、`draft_block_table`，与主池同步分配 | 两个模型**共用一张 block_table**，同一块号在两个池中各自寻址 | 按它的显存划分公式，两个池的块数本来就相等，两套管理维护的是同一份信息。共用之后 BlockManager 不需要知道草稿模型的存在，而且**草稿模型自动获得 prefix cache**（原项目的草稿池不做 prefix cache，命中前缀时仍要重算整个 prompt） | 五.4 |
| 2 | **草稿模型如何追上进度** | 用 `draft_kv_len` 记录落后多少；只要 batch 中有一条序列落后，第 0 步就**整批**走 eager 的 varlen prefill | prefill 时两个模型一起算；decode 时第 0 步**固定输入最后 2 个 token**（Q=2） | 不需要额外字段，第 0 步的形状永远是 `(B, 2)`，可以录 CUDA Graph。原项目中新序列加入、或某条序列 K 个全部被接受时，都会触发 eager 路径 | 五.4、五.5 |
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

- **Q=2 的第 0 步**：每一步都多算 1 个 token。对小模型来说影响不大，但确实比原项目"只在落后时追赶"多做了计算；
- **prefill 时两个模型一起算**：每次 prefill 都要多跑一遍草稿模型。原项目推迟到第一次推测时才追赶，如果请求很短、几乎没有 decode，原项目的做法反而更省；
- **整批统一决定是否推测**：比逐条序列决定要粗糙。只要一条序列接近 `max_model_len`，整批都会退回普通 decode。好处是 verify 的形状统一，补齐的位置一定落在已分配的块里；
- **共用 block_table**：有一个小的例外。如果序列恰好在"K 个全部接受"的那一步结束，最后一个位置的草稿 KV 永远不会补上，之后命中这个块的请求会读到旧数据。这只会降低接受率，输出依然无损（设计文档第五节第 5 小节）。

## 四、总结

原项目解决了"让小模型打草稿跑起来并且快"这个问题，尤其是 verify 能录 CUDA Graph，这一点设计文档直接沿用了。设计文档的改进主要集中在三个方面：

- **和 nano-vllm 现有架构的契合**：共用 block_table，沿用进度字段和 `hash_blocks`，由 Scheduler 统一修改状态（改进 1、3、4）；
- **边界情况**：TP、大 batch、`max_model_len`、词表校验（改进 5、6、7、10）；
- **CUDA Graph 的覆盖率和显存**：Q=2 的第 0 步、共享内存池（改进 2、8、12）。

这些改进是否真能带来收益，还要等实现后用实测数据来验证。

---

## 五、实现对照清单

按设计文档第七节第 3 小节的实现顺序排列，每完成一项打勾。括号中是对应的改进编号。

**第 1 步：配置与骨架**
- [ ] `SpeculativeConfig`：按方法校验、词表一致、token 预算断言（10）
- [ ] `Context.q_len` + attention decode 分支按 `q_len` reshape（12）
- [ ] `BlockManager.can_append(seq, n)` / `may_append(seq, n)`：预留 `len + K`，多余的块不释放（沿用）
- [ ] Scheduler decode 分支：按 batch 大小、`max_model_len` 整批决定是否推测（6、7）
- [ ] `postprocess_spec`：草稿不进 `token_ids`，先修正 `num_scheduled_tokens` 再 `hash_blocks`（3、4）
- [ ] 验证：`num_drafts = 0` 时输出分布与关闭推测解码时一致

**第 2 步：n-gram + 拒绝采样**
- [ ] `NgramProposer`（11）
- [ ] `RejectionSampler`：向量化写法，`q` 为 `None` 时按 one-hot 处理（沿用、11）
- [ ] verify 走 decode 路径（沿用）
- [ ] 验证：重复性强的 prompt 接受率明显大于 0

**第 3 步：草稿模型**
- [ ] 两个模型共用 block_table，`allocate_kv_cache` 按两个模型的块字节数之和分配（1）
- [ ] prefill 时草稿模型一起前向，warmup 自然计入草稿模型的激活（2、9）
- [ ] 草稿第 0 步固定 Q=2，之后 Q=1（2）
- [ ] 验证：草稿模型 = 大模型时，接受率接近 100%

**第 4 步：边界测试**
- [ ] 抢占、prefix cache 命中、`max_tokens` / EOS 出现在草稿中间、接近 `max_model_len`

**第 5 步：CUDA Graph**
- [ ] verify 图 + 两组草稿图，所有图共用一个内存池，先录最大的 verify 图（8）

**第 6 步：统计与调参**
- [ ] 按实际 token 数统计吞吐，打印接受率（13）
- [ ] 在 B = 1、8、16、32、64 上实测，确定 `max_batch_size` 和 K 的默认值（6）

**第 7 步（可选）**
- [ ] TP > 1：`run_spec` 在所有 rank 上执行，草稿 token `dist.broadcast`（5）
