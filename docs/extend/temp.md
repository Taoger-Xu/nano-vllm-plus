# 推测解码实现进度（临时记录）

记录当前推测解码的实现进度、已发现的问题和下一步计划。完整设计见 [SpeculativeDecoding.md](SpeculativeDecoding.md)，与参考项目的对比和完整的实现清单见 [spdraft.md](spdraft.md)。

> 更新时间：2026-10-17。以下"已完成"均为代码层面完成，只做了 CPU 测试，尚未在 GPU 上端到端运行。

---

## 一、架构调整：草稿模型的 KV 改为两套独立管理

最初的设计（SpeculativeDecoding.md 五.4、五.5）是**两个模型共用一张 block_table**，草稿模型的进度不单独记录。现已改为与参考项目 B 相同的思路：**两套 KVState、两个 BlockManager，各自分配、各自记录进度**。

| | 原设计：共用 block_table | 现在：两套独立管理 |
|---|---|---|
| 块的分配 | 一张表，天然一致 | 两个 BlockManager 同进同退：准入、预留要两边都放得下，抢占、结束时两边一起释放 |
| 草稿模型的进度 | 隐式：靠"大模型写了哪里，草稿模型也写了"这一前提 | **显式**：`seq.draft_kv.num_cached_tokens` |
| 不推测的 decode 步 | 必须也运行草稿模型，否则出现空洞（原 B-1） | **草稿模型不运行，允许落后**；下次推测时按 `draft_kv.num_cached_tokens` 一次补上 |
| prefix cache | 两个模型都能用 | **开启草稿模型时关闭**（`Config` 自动设置 `enable_prefix_cache = False`）。大模型命中前缀时两个模型的 prefill 起点会不同，第一版不处理 |
| prefill 的输入 | 两个模型共用一个 Context | 草稿模型用 `prepare_prefill(seqs, "draft")` 单独生成 `slot_mapping` / `block_tables` |
| 显存划分 | 按两个模型每块字节数之和计算，块数相同 | 不变：两个池块数相同（`num_kvcache_blocks` = `num_draft_kvcache_blocks`），块号空间各自独立 |

带来的变化：

- **原问题 1（草稿 KV 空洞）和 B-1 不再需要处理**：落后变成了显式、可补齐的状态，不推测的步也不必额外运行草稿模型；
- **原"Q=2 会写入已登记的共享块"的问题也不存在了**：草稿模型不做 prefix cache，它的块不会被其他请求共享；
- **代价**：开启草稿模型时失去 prefix cache；prefill 时 CPU 上要多准备一份草稿模型的输入。

## 二、已完成

| 内容 | 位置 | 对应设计 |
|---|---|---|
| `SpeculativeConfig`：按方法校验；`Config` 中断言 token 预算、TP = 1、两个模型词表一致 | [config.py](../../nanovllm/config.py) | 三.2 |
| `Config.enable_prefix_cache`（开启草稿模型时自动关闭）、`Config.num_draft_kvcache_blocks` | config.py | 第一节 |
| `KVState`：`Sequence` 持有 `target_kv` / `draft_kv`，`seq.kv(role)` 按角色取；原有的 `block_table` / `num_cached_tokens` / `num_scheduled_tokens` 改为转发到 `target_kv` 的属性；跨进程序列化带上两份状态 | [sequence.py](../../nanovllm/engine/sequence.py) | 第一节 |
| `BlockManager(role, enable_prefix_cache)`：所有方法只读写 `seq.kv(role)`；关闭 prefix cache 时不查表、不登记 hash | [block_manager.py](../../nanovllm/engine/block_manager.py) | 第一节 |
| `can_append(seq, n)` / `may_append(seq, n)`：按"覆盖到 `num_cached_tokens + n`"计算新块数，多留的块不重复分配；n = 0 表示本步不写 | block_manager.py `_num_new_blocks` | 五.7 |
| Scheduler：开启草稿模型时创建第二个 BlockManager，两边同进同退；decode 分支 `_decode_num_tokens` 整批决定 n，`_draft_num_tokens` 计算草稿模型本步要写多少（不推测时为 0，推测时补齐落后部分 + K 个草稿） | [scheduler.py](../../nanovllm/engine/scheduler.py) | 五.8、第一节 |
| 加载草稿模型；两个池按每块字节数之和划分显存，各自绑定 | [model_runner.py](../../nanovllm/engine/model_runner.py) `allocate_kv_cache`、`bind_kv_cache` | 五.4 |
| `prepare_prefill` / `prepare_block_tables` 增加 `role`；`run_prefill_models` 先在大模型的 Context 下算完 logits，再为草稿模型单独准备输入并前向 | model_runner.py | 第一节 |
| `prepare_decode` 的 slot 改为按位置查块（不再取 `block_table[-1]`），推测步之后多留了块也能写对位置 | model_runner.py | 五.7 |
| `Proposer` 接口、`NgramProposer`、`DraftModelProposer`（框架） | [spec_decode/](../../nanovllm/spec_decode/) | 五.1、五.2 |
| 临时保护：`Scheduler.spec_enabled = False`，n 目前恒为 1；`LLMEngine.step()` 遇到 n > 1 直接抛出 `NotImplementedError` | scheduler.py、[llm_engine.py](../../nanovllm/engine/llm_engine.py) | — |
| 注释风格统一：方法用 `"""` 写功能和参数，步骤注释放在代码上一行 | sequence.py、block_manager.py、scheduler.py、model_runner.py | — |

**CPU 测试**（脚本在本地 scratchpad，没有加入仓库；用 stub 代替 `xxhash`、flash-attn、triton、`torch.compile`）：

- `can_append` / `may_append`：n = 1 时与原逻辑逐一相同（约 22 万个 decode 步），n > 1 跨块、多留块正确；
- `Sequence` / `KVState`：转发属性、两份状态互相独立、pickle 往返；
- 两个角色的 BlockManager：块号空间独立、草稿模型不登记 hash、前缀只在大模型一侧命中、按角色预留和释放；
- Scheduler（开启草稿模型）：chunked prefill 时两边进度相等；不推测时草稿模型落后；推测时草稿模型的预留覆盖"补齐 + K"；只有草稿池不够时触发抢占且两边一起释放；结束时两个池完全归还；
- ModelRunner：两个角色的 `slot_mapping` 分别对应各自的块表；`run_prefill_models` 在大模型的 Context 下算 logits、在草稿模型的 Context 下运行草稿模型；`prepare_decode` 与原公式一致。

## 三、已发现、尚未处理的问题

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| 1 | `DraftModelProposer` 只做了 K 次 Q=1 的前向，没有处理草稿模型的落后。现在落后的长度是可变的（不推测的步不运行草稿模型） | 草稿模型落后时猜不准 | 第 0 步输入 `[draft_kv.num_cached_tokens, len(seq))` 这一段（长度可变，走 prefill 式的输入准备，CUDA Graph 下需要补齐或分档），之后 K−1 步为 Q=1 |
| 2 | `warmup_model` 不经过 `run()`，跳过了 `compute_logits` 和 `Sampler` | `Sampler` 的 `torch.compile` 推迟到第一个真实请求，首个请求变慢 | warmup 末尾补一次 logits 和采样 |
| 3 | 草稿 KV cache 按大模型的默认 dtype 分配，但字节数按草稿模型的 dtype 计算 | 两个模型 dtype 不同时显存估算不准（Qwen3 都是 bf16，暂无影响） | `Config` 中断言两个模型 dtype 一致 |
| 4 | `_decode_num_tokens` 用 `len(self.running)` 判断 batch 大小，实际调度数还受 `max_num_seqs` 限制 | `max_num_seqs ≤ max_batch_size` 时可能误关推测（默认配置下不会发生） | 改为 `min(len(self.running), self.max_num_seqs)`（可选） |
| 5 | 草稿模型落后很多时，一次补齐的计算量可能很大（例如 batch 长期超过 `max_batch_size` 后再降下来） | 重新进入推测的那一步变慢 | 观察实测数据；必要时限制单步补齐的长度，或给推测开关加滞后 |
| 6 | 小问题：`NgramProposer` 中多余的 `min`、不必要的 `inference_mode`、非锁页内存的 `non_blocking` 拷贝；`Scheduler.__init__` 文档字符串的 `Args` 中缺少参数名 `config (Config):` | 无功能影响 | 顺手清理 |
| 7 | [SpeculativeDecoding.md](SpeculativeDecoding.md) 五.4、五.5 和 [spdraft.md](spdraft.md) 仍按"共用 block_table"描述 | 文档与代码不一致 | 按第一节更新设计文档；spdraft.md 中 B-1 标记为不再需要 |

## 四、设计评估的结论

- **正确性**：两套独立管理的方案同样是无损的。草稿模型的 KV 对不对只影响接受率；大模型的 KV、hash 登记、拒绝采样都不受影响。
- **一致性**：块分配有不一致的风险（一边分配了、另一边没分配），所以 Scheduler 中所有准入、预留、释放都必须两边同时处理；进度的不一致变成了显式的，可以在每轮推测开始时断言 `draft_kv.num_cached_tokens ≤ len(seq) − 1`。
- **开启草稿模型后 KV 容量约降为原来的 56%**（Qwen3-0.6B 每 token 的 KV 约为 Qwen3-8B 的 78%）。这是保留完整草稿 KV 的固有代价；实测 `max_batch_size` 时要同时观察抢占次数。
- **推测开关可加滞后**，避免 batch 在阈值附近时频繁切换，同时减少问题 5 中大段补齐的次数。

## 五、下一步

按 [spdraft.md](spdraft.md) 第九节的顺序，当前处于第 1 步（骨架）的后半段：

1. `postprocess_spec`：逐个追加被接受的草稿 → 把大模型的 `num_scheduled_tokens` 改成确认数 → `hash_blocks` → 推进两个模型各自的进度（草稿模型按本轮实际写入的位置）→ 追加修正 token（设计五.6）
2. `Context.q_len` 和 attention decode 分支按 `q_len` reshape（五.9 (2)）；或采用 varlen 路径录图的备选方案（spdraft.md B-4）
3. `prepare_verify`、`run_spec`、`RejectionSampler`（五.3、五.9）
4. `LLMEngine.step()` 接上 `run_spec` / `postprocess_spec`，打开 `spec_enabled`
5. 先用"草稿全部拒绝"跑通，确认输出与普通 decode 同分布；再接入 n-gram，最后接入草稿模型并处理问题 1
6. 按第一节更新设计文档（问题 7）
