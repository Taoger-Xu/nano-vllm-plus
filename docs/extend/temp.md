# 推测解码实现进度（临时记录）

记录当前推测解码的实现进度、已发现的问题和下一步计划。完整设计见 [SpeculativeDecoding.md](SpeculativeDecoding.md)，与参考项目的对比和完整的实现清单见 [spdraft.md](spdraft.md)。

> 更新时间：2026-10-17。以下"已完成"均为代码层面完成，尚未在 GPU 上端到端运行。

---

## 一、已完成

| 内容 | 位置 | 对应设计 |
|---|---|---|
| `SpeculativeConfig`：按方法校验；`Config` 中断言 token 预算、TP = 1、两个模型词表一致 | [config.py](../../nanovllm/config.py) | 三.2 |
| 加载草稿模型；两个模型共用 block_table，KV cache 按两个模型每块字节数之和分配，各自绑定 | [model_runner.py](../../nanovllm/engine/model_runner.py) `__init__`、`allocate_kv_cache`、`bind_kv_cache` | 五.4 |
| prefill 时草稿模型用同一份输入一起前向（只写 KV）；warmup 中两个模型都执行 | `run_prefill_models`、`warmup_model` | 五.4 |
| `Proposer` 接口、`NgramProposer`、`DraftModelProposer`（框架） | [spec_decode/](../../nanovllm/spec_decode/) | 五.1、五.2 |
| `can_append(seq, n)` / `may_append(seq, n)`：按"覆盖到 `num_cached_tokens + n`"计算新块数，多留的块不重复分配 | [block_manager.py](../../nanovllm/engine/block_manager.py) `_num_new_blocks` | 五.7 |
| Scheduler decode 分支：`_decode_num_tokens` 整批决定 n（未开启 / batch 超过 `max_batch_size` / 接近 `max_model_len` 时为 1），按 n 检查、抢占、分配，设置 `num_scheduled_tokens = n` | [scheduler.py](../../nanovllm/engine/scheduler.py) | 五.8 |
| 临时保护：`Scheduler.spec_enabled = False`，n 目前恒为 1；`LLMEngine.step()` 遇到 n > 1 直接抛出 `NotImplementedError` | scheduler.py、[llm_engine.py](../../nanovllm/engine/llm_engine.py) | — |
| 注释风格统一：方法用 `"""` 写功能和参数，步骤注释放在代码上一行 | block_manager.py、scheduler.py、model_runner.py | — |

**验证**：在 CPU 上对比了新旧 `can_append` / `may_append`（block_size = 256、512，约 22 万个 decode 步），n = 1 时结果逐一相同；n > 1 跨块、多留块的情况也检查通过。测试脚本用一个 stub 代替 `xxhash`，没有加入仓库。

## 二、已发现、尚未处理的问题

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| 1 | 不推测的 decode 步只运行大模型，草稿模型的 KV 会出现空洞；登记了 hash 的块会把空洞扩散给命中它的请求 | 只影响接受率，不影响正确性 | 不推测的步也运行一次草稿模型（Q=1，只写 KV）。即 spdraft.md 的 B-1 |
| 2 | `DraftModelProposer` 没有实现第 0 步 Q=2；直接改 `prepare_draft_decode` 返回 Q=2 会让 `positions + 1` 的形状出错 | 草稿模型落后 1 个 token 时补不上 | 第 0 步单独处理，之后 K−1 步为 Q=1 |
| 3 | `warmup_model` 不再经过 `run()`，跳过了 `compute_logits` 和 `Sampler` | `Sampler` 的 `torch.compile` 推迟到第一个真实请求，首个请求变慢 | warmup 末尾补一次 logits 和采样 |
| 4 | 草稿 KV cache 按大模型的默认 dtype 分配，但字节数按草稿模型的 dtype 计算 | 两个模型 dtype 不同时显存估算不准（Qwen3 都是 bf16，暂无影响） | `Config` 中断言两个模型 dtype 一致 |
| 5 | `_decode_num_tokens` 用 `len(self.running)` 判断 batch 大小，实际调度数还受 `max_num_seqs` 限制 | `max_num_seqs ≤ max_batch_size` 时可能误关推测（默认配置下不会发生） | 改为 `min(len(self.running), self.max_num_seqs)`（可选） |
| 6 | 小问题：`NgramProposer` 中多余的 `min`、不必要的 `inference_mode`、非锁页内存的 `non_blocking` 拷贝 | 无功能影响 | 顺手清理 |

## 三、设计评估的结论

- **正确性**：共用 block_table + 保留 prefix cache 的方案在所有情况下都是无损的；大模型的 KV 和 hash 登记是安全的。
- **Q=2 会写入已登记、可能被共享的块**（位置 `len−2`）。写入的值与原来相同，无害，但打破了"只往未写满的块中写"的惯例，需要在文档和代码注释中说明。
- **开启草稿模型后 KV 容量约降为原来的 56%**（Qwen3-0.6B 每 token 的 KV 约为 Qwen3-8B 的 78%）。任何保留完整草稿 KV 的方案都有这个代价；实测 `max_batch_size` 时要同时观察抢占次数。
- **问题 1 的修正（B-1）不是免费的**：不推测的步每步多一次小模型前向，约 +10%~20%。第一版先这样实现，再实测对比"不运行草稿模型"的方案。
- **建议增加显式的草稿进度字段**（如 `draft_num_cached_tokens`），每轮断言"最多落后 1 个"，把隐式依赖变成可检查的不变量。
- **推测开关可加滞后**，避免 batch 在阈值附近时频繁切换。

## 四、下一步

按 [spdraft.md](spdraft.md) 第九节的顺序，当前处于第 1 步（骨架）的后半段：

1. `postprocess_spec`：逐个追加被接受的草稿 → 把 `num_scheduled_tokens` 改成确认数 → `hash_blocks` → 推进 `num_cached_tokens` → 追加修正 token（设计五.6）
2. `Context.q_len` 和 attention decode 分支按 `q_len` reshape（五.9 (2)）；或采用 varlen 路径录图的备选方案（spdraft.md B-4）
3. `prepare_verify`、`run_spec`、`RejectionSampler`（五.3、五.9）
4. `LLMEngine.step()` 接上 `run_spec` / `postprocess_spec`，打开 `spec_enabled`
5. 先用"草稿全部拒绝"跑通，确认输出与普通 decode 同分布；再接入 n-gram，最后接入草稿模型并处理问题 1、2
