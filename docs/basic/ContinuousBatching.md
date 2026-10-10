# nano-vllm 中的 Continuous Batching

本文分三部分：
1. 为什么需要 continuous batching；
2. 它依赖的三项配套技术，以及各自的必要性；
3. nano-vllm 的具体实现。

文中的代码链接都指向本仓库的源码。

---

## 一、为什么需要 Continuous Batching

### 1. LLM 推理的两个阶段

- **Prefill**：一次性处理整个 prompt，属于计算密集（compute-bound）。
- **Decode**：每步只生成 1 个 token，但每一层都要把整套权重读一遍，属于访存密集（memory-bound）。只跑 1 条序列时 GPU 算力大部分空着。**batch 越大，一次读权重分摊到的 token 越多**，吞吐几乎线性上涨。

所以 decode 阶段必须攒 batch。问题是怎么攒。

### 2. 传统 Static Batching 的问题

静态批处理的做法是：凑齐 N 条请求组成一个 batch，一直跑到**所有**请求都结束，再处理下一批。

```
时间 →        t1  t2  t3  t4  t5  t6  t7  t8
seq A (短)    ██  ██  ██  EOS ··  ··  ··  ··   ← 结束后槽位空转
seq B (长)    ██  ██  ██  ██  ██  ██  ██  EOS
seq C (中)    ██  ██  ██  ██  ██  EOS ··  ··   ← 空转
新请求 D      等待…………………………………………………………  ← 必须等整批结束
```

它有几个问题：

1. **输出长度差别很大**：同一个 batch 里，有的回复 10 个 token，有的 2000 个。短序列结束后，它的槽位一直空着，实际 batch size 越跑越小，GPU 利用率随之下降。
2. **队头阻塞，延迟高**：新请求要等整批里最长的那条跑完才能进入。
3. **Padding 浪费**：长度不同的序列要 pad 成矩形张量，pad 部分的计算和显存都白费。
4. **KV cache 要按最大长度预留**：显存利用率低，能放下的并发序列数少。

### 3. Continuous Batching 的思路

Continuous batching 也叫 iteration-level scheduling：**调度粒度从"一个请求"变成"一次前向（一个 step）"**。

- 每个 step 之后，已结束的序列**立即退出**，释放资源；
- 等待队列里的新序列**下一个 step 就能加入**；
- batch 的成员每一步都在变化，GPU 始终保持较大的有效 batch。

---

## 二、三项配套技术

Continuous batching 的核心是：**每个 step 的 batch 成员都在变化**。序列可以在任意时刻加入或退出，同一个 batch 里的序列长度、进度也都各不相同。

这种动态性会带来三个问题，三项配套技术分别解决其中一个：

```
batch 成员每步都在变
   ├─ 问题1：各序列长度不同、每步都变 → 无法拼成规整的矩形张量 → 变长 attention（去 padding）
   ├─ 问题2：显存要频繁分配和释放，且不知道每条序列最终会多长 → PagedAttention
   └─ 问题3：问题2 采用"按需分配、乐观接纳"后，显存可能中途耗尽 → 抢占
```

### 1. 变长 attention：解决"形状不规整"

**问题**：静态批处理里，一批序列一起开始、一起推进，长度差异有限，pad 到最长还能接受。continuous batching 下差异会被极度放大。比如同一个 decode step 里：

- 序列 A 已经跑了很久，上下文长 3000；
- 序列 B 刚加入，上下文长 20。

如果组成 `[batch, max_len]` 的矩形张量，B 要 pad 近 3000 个无效位置。batch 里新老序列越混杂，浪费越大。而且每步的 max_len 都在变，张量形状也跟着变。

**解法**：不 pad，把所有序列的 token 首尾拼成一维，再用"边界数组"告诉 kernel 每条序列从哪里开始、到哪里结束。

```
A 有 5 个 token，B 有 3 个 token
input_ids    = [a1 a2 a3 a4 a5 b1 b2 b3]   ← 一维，没有 pad
cu_seqlens_q = [0, 5, 8]                    ← A 占 [0,5)，B 占 [5,8)
```

`flash_attn_varlen_func` 根据 `cu_seqlens` 只在每条序列内部计算 attention（[model_runner.py `prepare_prefill`](../../nanovllm/engine/model_runner.py#L129-L170)）。decode 时则用 `context_lens` 记录每条序列各自的上下文长度（[model_runner.py `prepare_decode`](../../nanovllm/engine/model_runner.py#L172-L188)）。这样计算量只与真实 token 数成正比，跟谁长谁短无关。

### 2. PagedAttention：解决"显存怎么分"

**问题**：每条序列都要占一块 KV cache，大小随生成的 token 数增长。但**事先不知道它会生成多少个 token**。

传统做法是给每条序列**连续地**预留 `max_model_len` 那么多空间。

举例：剩余显存能放 25600 个 token 的 KV。

- 连续预留，每条按 4096 算：只能同时跑 **6 条**。如果实际平均每条只用 600 个 token，约 85% 的显存被预留了却没用上。
- 序列不停地进进出出，连续分配还会产生碎片。这和 `malloc` 的情况一样：总空闲够，但没有一段连续空间放得下。

batch 开不大，continuous batching 的收益也就没有了。

**解法**：借鉴操作系统的分页机制。

- 把 KV cache 切成固定大小的 block（nano-vllm 中每块 256 个 token），见 [model_runner.py `allocate_kv_cache`](../../nanovllm/engine/model_runner.py#L103-L121)。
- 每条序列维护一张 `block_table`（逻辑块号 → 物理块号），物理块**不需要连续**，任何空闲块都能用。
- **按需分配**：prefill 时只分配 prompt 需要的块；decode 时只在序列写满当前块、跨入新块时才多分配一个（[block_manager.py `may_append`](../../nanovllm/engine/block_manager.py#L106)）。
- **结束立即归还**：序列结束后 `deallocate`，空出的块下一个 step 就能给新序列用。

同一个例子：每条序列最多浪费不到一个块，平均 600 个 token 只占 3 个块，25600 个 token 能放下**约 30 条**序列，并发数提升了好几倍。

另一个好处是，attention kernel 通过 `block_table` 找到每条序列自己的 KV（[attention.py](../../nanovllm/layers/attention.py#L72)），**KV 不再绑定"batch 里第几行"**。所以序列在 batch 里的位置每步都可以变，有序列退出也不需要搬动其他序列的数据。

### 3. 抢占：解决"乐观接纳后显存可能不够"

**问题**：这是第 2 点带来的副作用。看接纳新序列时的检查（[block_manager.py `can_allocate`](../../nanovllm/engine/block_manager.py#L58-L73)），它只检查**当前 prompt**需要的块够不够，没有为将来要生成的 token 预留空间。这是一种**乐观接纳**（overcommit）：先让尽量多的序列跑起来，赌它们不会同时变得很长。

但赌输的情况是会发生的。所有 running 序列每步都在变长，某个时刻某条序列需要新块，而空闲块已经用完。静态批处理按最大长度预留，永远不会遇到这种情况；乐观接纳则必须有兜底方案。

**解法**：牺牲一部分序列，腾出空间。见 [scheduler.py](../../nanovllm/engine/scheduler.py#L60-L65)：

```python
while not self.block_manager.can_append(seq):   # 需要新块但没有空闲块
    if self.running:
        self.preempt(self.running.pop())        # 踢掉最晚加入的序列
    else:
        self.preempt(seq)                       # 只剩自己了，就抢占自己
```

[`preempt()`](../../nanovllm/engine/scheduler.py#L75) 的做法：释放被踢序列的所有块，把它放回 `waiting` **队头**，等有空间时**重新 prefill**（recompute）。重算会多花一点时间，但换来了平时高得多的并发。另外，如果它原来的块还没被覆盖，重新 prefill 时还能命中 prefix cache，少算一部分。

### 用操作系统来类比

| LLM 推理 | 操作系统 |
|---|---|
| continuous batching：序列随时进出 | 多进程调度：进程随时创建、退出 |
| PagedAttention：block + block_table | 虚拟内存分页：page + 页表 |
| 乐观接纳 | 内存超分配（overcommit） |
| 抢占 + recompute | 内存不足时换出进程 |
| 变长 attention | 让每个进程只用它真正需要的资源，不按最大规格对齐 |

小结：continuous batching 让 batch 变成**动态**的；变长 attention 解决了**计算层面**的动态，PagedAttention 解决了**显存层面**的动态，抢占则是 PagedAttention 采用乐观分配之后必须有的**兜底机制**。三者缺一个，continuous batching 要么跑不起来，要么 batch 开不大、收益很小。

---

## 三、nano-vllm 的实现

整体结构如下：

```
LLMEngine.generate()  ──循环──►  step()
                                   ├─ Scheduler.schedule()     决定本 step 跑哪些 seq
                                   ├─ ModelRunner.run()        拼成扁平 batch，前向
                                   └─ Scheduler.postprocess()  追加 token、回收结束的 seq
```

### 1. 以 step 为单位的主循环

[llm_engine.py `step`](../../nanovllm/engine/llm_engine.py#L49-L55)：

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()      # 每步重新挑选 batch
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    outputs = [... for seq in seqs if seq.is_finished]  # 结束的立即返回
```

`generate()` 中的 `while not self.is_finished()`（[llm_engine.py](../../nanovllm/engine/llm_engine.py#L73)）不断调用 `step()`。**每次 step 都重新调度一次**，这就是 iteration-level scheduling。

### 2. 序列状态机：waiting / running 两个队列

[sequence.py](../../nanovllm/engine/sequence.py) 中每条请求是一个 `Sequence`，状态有 `WAITING → RUNNING → FINISHED` 三种。调度器（[scheduler.py](../../nanovllm/engine/scheduler.py#L16-L17)）维护两个队列：

- `waiting`：还没完成 prefill 的序列（新请求、被抢占的请求）
- `running`：已完成 prefill、正在 decode 的序列

另外有两个字段配合 chunked prefill 使用：`num_cached_tokens` 表示已写进 KV cache 的 token 数，`num_scheduled_tokens` 表示本 step 要算的 token 数。

### 3. 调度策略 `schedule()`

[scheduler.py `schedule`](../../nanovllm/engine/scheduler.py#L25-L73) 采用 **prefill 优先**的策略，**同一个 step 里不混合 prefill 和 decode**。

**(a) Prefill 阶段**（[第 29-55 行](../../nanovllm/engine/scheduler.py#L29-L55)）

- 从 `waiting` 队头依次取序列，受两个预算约束：`max_num_seqs`（默认 512）和 `max_num_batched_tokens`（默认 16384）。
- `block_manager.can_allocate()` 检查 KV block 是否足够（同时计算 prefix cache 命中了多少 block）。不够就停止接纳新序列。
- **Chunked prefill**（[第 42 行](../../nanovllm/engine/scheduler.py#L42)）：token 预算不够时，**只有 batch 里的第一条序列**可以只算一部分（`min(num_tokens, remaining)`）。剩余部分留在 `waiting` 队头，下个 step 接着算。这样超长 prompt 不会卡住调度。
- prompt 全部算完的序列转入 `running`。
- 只要本轮调度到了任何 prefill，就直接返回 `is_prefill=True`。

**(b) Decode 阶段**（[第 57-73 行](../../nanovllm/engine/scheduler.py#L57-L73)）

- 没有待 prefill 的序列时，从 `running` 中取序列，每条本步只算 1 个 token。
- `can_append()` 检查这条序列跨入新 block 时是否还有空闲 block，不够就触发上文所述的抢占。

### 4. 结束即回收：`postprocess()`

[scheduler.py `postprocess`](../../nanovllm/engine/scheduler.py#L81-L92)：

- 更新 `num_cached_tokens`。chunked prefill 还没算完的序列直接 `continue`，不追加 token。
- 追加采样到的 token；遇到 EOS 或达到 `max_tokens` 时，序列标记为 `FINISHED`，**立即 `deallocate` 释放 KV block**，并从 `running` 移除。
- 释放的 block 在下一个 step 就能被 `waiting` 里的新序列使用。continuous batching 里"有序列走、就有新序列补上"的效果靠的就是这一步。

### 5. 扁平化 batch 与 Context 传递

`ModelRunner` 把调度结果整理成一维的 `input_ids` / `positions`，以及 `cu_seqlens`、`slot_mapping`、`context_lens`、`block_tables` 等元数据。元数据通过全局 [Context](../../nanovllm/utils/context.py) 传进每一层的 `Attention`，模型代码本身不需要知道 batch 的结构。`slot_mapping` 告诉 [store_kvcache](../../nanovllm/layers/attention.py#L33) 这个 Triton kernel，每个新 token 的 K/V 写到物理 cache 的哪个位置。

### 6. Prefix caching

[block_manager.py](../../nanovllm/engine/block_manager.py) 按 block 做链式 hash（`hash_blocks` / `can_allocate`），相同前缀的 block 通过 `ref_count` 在序列间共享，减少重复 prefill。被抢占的序列重新 prefill 时也能从中受益。

### 7. 适配不断变化的 batch size：CUDA Graph 分桶

decode 阶段 batch size 每步都可能不同，而 CUDA Graph 要求输入形状固定。`capture_cudagraph`（[model_runner.py](../../nanovllm/engine/model_runner.py#L234)）预先为 `bs = 1, 2, 4, 8, 16, 32, …, 512` 各录制一张图。运行时（[`run_model`](../../nanovllm/engine/model_runner.py#L197-L212)）选**不小于当前 bs 的最小一档**，多出的位置用 `slot_mapping = -1` 填充，kernel 遇到 -1 会直接跳过写入。prefill 每步长度变化太大，所以走 eager 模式。

---

## 四、小结：与 vLLM 的差异

| 方面 | nano-vllm 的做法 |
|---|---|
| 调度粒度 | 每个 step 重新调度（iteration-level） |
| prefill 与 decode | **不混批**，prefill 优先；有待 prefill 的序列时，decode 暂停一步 |
| 长 prompt | chunked prefill，但只允许 batch 中第一条序列被切块 |
| 显存不足 | recompute 式抢占（LIFO，踢最晚加入的），没有 swap |
| KV 管理 | PagedAttention + prefix cache（hash + 引用计数） |
| 去 padding | 拼成扁平 batch + FlashAttention varlen / kvcache 接口 |
| batch size 变化 | CUDA Graph 分桶 |

和 vLLM v1 / Sarathi-Serve 的主要区别是：**nano-vllm 没有在同一个 step 里把 prefill 块和 decode token 混在一起**。所以新请求进来时，正在 decode 的序列会被打断一步（生成 token 的间隔出现抖动）。好处是调度逻辑非常简洁，整个调度器只有约 90 行，适合用来理解 continuous batching 的核心机制。
