# Block 与 BlockManager：使用场景、Chunked Prefill 与显存安全

本文从**使用场景**的角度梳理 [block_manager.py](../nanovllm/engine/block_manager.py)：每个函数在什么时候被调用、为什么被调用。然后介绍与块分配密切相关的 chunked prefill，并回答"分块计算会不会让显存溢出"这个问题。

每个函数内部的逐行分析、块的状态机和完整状态追踪，见 [PagedAttention.md](PagedAttention.md)。本文不重复那些内容。

---

## 一、函数分组

```
BlockManager
├─ 对外接口（只有 Scheduler 调用）
│   ├─ can_allocate / allocate      ← prefill 准入
│   ├─ can_append / may_append      ← decode 扩容
│   ├─ hash_blocks                  ← 每步结束后登记
│   └─ deallocate                   ← 释放
├─ 内部工具
│   ├─ _allocate_block / _deallocate_block   ← 真正从空闲队列取块、还块
│   └─ compute_hash                          ← 计算链式 hash
Block
├─ update   ← 只被 hash_blocks 调用（登记 hash）
└─ reset    ← 只被 _allocate_block 调用（块被重新分配）
```

**BlockManager 只有 Scheduler 一个调用方**（[scheduler.py](../nanovllm/engine/scheduler.py)）。ModelRunner 不会调用它，只读取 `seq.block_table` 和 `num_cached_tokens` 这两个结果。

---

## 二、场景总览

| # | 场景 | 触发的函数 | 调用位置 |
|---|---|---|---|
| 0 | 系统启动 | `BlockManager.__init__` | [scheduler.py:15](../nanovllm/engine/scheduler.py#L15) |
| 1 | 新请求第一次 prefill | `can_allocate` → `allocate` → `_allocate_block` × n | [scheduler.py:36](../nanovllm/engine/scheduler.py#L36)、[:45](../nanovllm/engine/scheduler.py#L45) |
| 2 | 新请求命中前缀 | 同上，部分块改为共享或从空闲队列中捞回 | 同上 |
| 3 | 空闲块不够，接纳不了 | `can_allocate` 返回 -1 | [scheduler.py:37-38](../nanovllm/engine/scheduler.py#L37-L38) |
| 4 | chunked prefill 的后续块 | 不分配块，只调用 `hash_blocks` | [scheduler.py:41](../nanovllm/engine/scheduler.py#L41)、[:83](../nanovllm/engine/scheduler.py#L83) |
| 5 | 每一步结束 | `hash_blocks` → `Block.update` | [scheduler.py:83](../nanovllm/engine/scheduler.py#L83) |
| 6 | decode，没跨入新块 | `can_append`（直接通过）→ `may_append`（什么都不做） | [scheduler.py:60](../nanovllm/engine/scheduler.py#L60)、[:69](../nanovllm/engine/scheduler.py#L69) |
| 7 | decode，跨入新块 | `can_append` → `may_append` → `_allocate_block` | 同上 |
| 8 | decode 时空闲块不够 | `can_append` 失败 → 抢占：`deallocate(被抢占者)` → 再次 `can_append` | [scheduler.py:60-65](../nanovllm/engine/scheduler.py#L60-L65)、[:78](../nanovllm/engine/scheduler.py#L78) |
| 9 | 序列正常结束 | `deallocate` → `_deallocate_block` | [scheduler.py:91](../nanovllm/engine/scheduler.py#L91) |
| 10 | 被抢占的序列重新调度 | 回到场景 1 或 2 | 同场景 1 |
| 11 | 旧缓存被覆盖 | `_allocate_block` 中清除旧 hash（被动发生） | 场景 1、7 内部 |

### 场景 0：系统启动

```
ModelRunner 测量显存 → config.num_kvcache_blocks = N
Scheduler.__init__ → BlockManager(N, block_size)
```

此时所有块都在空闲队列里，`hash_to_block_id` 为空。

需要注意：**warmup 不经过 BlockManager**。ModelRunner 在 warmup 时直接构造没有 `block_table` 的 Sequence，所以 BlockManager 从创建开始就是干净的状态。

### 场景 1：新请求第一次 prefill（没有命中前缀）

```
schedule() prefill 分支，seq.block_table 为空
  ├─ can_allocate(seq)      → 返回 0（没有命中，空闲块够用）
  └─ allocate(seq, 0)       → 为整个 prompt 调用 _allocate_block，每块一次
                               └─ Block.reset
```

- **时机**：序列位于 `waiting` 队头，第一次被调度。
- **结果**：`block_table` 被填满，`num_cached_tokens = 0`，整个 prompt 都要计算。

### 场景 2：新请求命中前缀

调用的函数和场景 1 相同，区别在 `allocate` 内部对命中块的处理：

| 命中块当前的状态 | 处理方式 | 典型例子 |
|---|---|---|
| 正被其他序列使用 | `ref_count += 1`，共享 | 多个请求同时在跑，用的是同一个 system prompt |
| 已释放，但还在空闲队列中 | 从空闲队列中取回 | 多轮对话：上一轮的请求刚结束，下一轮就来了 |

**结果**：`num_cached_tokens = 命中块数 × block_size`，prefill 只计算后面的部分。

### 场景 3：空闲块不够，接纳不了新请求

```
can_allocate(seq) → -1
  → schedule() 停止接纳新请求（break），seq 留在 waiting 队头
```

不会发生任何分配。如果这一步一条 prefill 也没调度到，调度器就转去执行 decode。等有序列结束、释放了块，下次 `schedule()` 再试。

### 场景 4：chunked prefill 的后续块

```
schedule() prefill 分支，seq.block_table 不为空
  → 跳过 can_allocate / allocate（第一次就已为整个 prompt 分配好了）
  → 只计算 num_scheduled_tokens
postprocess()
  → hash_blocks(seq)        登记本次 chunk 写满的块
```

块是在第一个 chunk 时**一次性分配好的**，后续 chunk 只管计算和登记，不再分配。详见第四节。

### 场景 5：每一步结束（prefill 和 decode 都会经过）

```
postprocess()，对本步调度的每条序列：
  hash_blocks(seq)  →  对本步刚写满的每一块调用 Block.update(h, token_ids)，并登记到 hash_to_block_id
```

这是**唯一往 prefix cache 中添加内容的入口**。

- prefill 一次可能写满多个块；
- decode 每 block_size 步才写满一块，其余步都是什么都不做直接返回。

prompt 中的 token 和生成出来的 token 都会经过这里登记，所以后续的请求可以复用前面请求生成的内容。

### 场景 6：decode，没跨入新块（绝大多数 decode 步）

```
can_append(seq)   → True（这一步不需要新块）
may_append(seq)   → 什么都不做
```

当 block_size = 256 时，每 256 个 decode 步中有 255 步属于这种情况。

### 场景 7：decode，跨入新块

```
can_append(seq)   → 检查至少还有 1 个空闲块
may_append(seq)   → _allocate_block → 追加到 block_table 末尾
```

**时机**：上一步生成的 token 正好要写到一个新块的第一个位置，即 `len(seq) % block_size == 1`。

### 场景 8：decode 时空闲块不够，触发抢占

```
can_append(seq) → False
  └─ preempt(running 队尾的序列)            （如果没有其他序列可抢，就抢占自己）
       └─ deallocate(被抢占者)
            └─ 对 ref_count 降为 0 的块调用 _deallocate_block
  └─ 再次 can_append(seq) → 成功后 may_append
```

**这是 decode 阶段唯一会释放别人块的路径。** 被抢占序列的块进入 FREE_CACHED 状态（hash 还在），为场景 10 留下了可以命中的缓存。

### 场景 9：序列正常结束

```
postprocess()：遇到 EOS 或达到 max_tokens
  └─ deallocate(seq)
       ├─ 所有块 ref_count -= 1
       └─ 降为 0 的块调用 _deallocate_block（放回空闲队列，hash 保留）
```

共享的前缀块只是引用计数减 1；只有最后一个使用者结束时，块才真正回到空闲队列。

### 场景 10：被抢占的序列重新调度

被抢占的序列已经回到 `waiting` 队头，`block_table` 也被清空，所以它会**重新走一遍场景 1 或 2**。如果它原来的块还没被覆盖，`can_allocate` 会命中这些块，只需重新计算没命中的部分。

### 场景 11：旧缓存被覆盖（被动发生）

这不是一个独立的调用，而是发生在 `_allocate_block` 内部：从空闲队列中取出一个处于 FREE_CACHED 状态的块时，先清除它的 hash 登记，然后才把它当作新块使用。

- **触发来源**：场景 1/2 中为未命中部分分配新块，或者场景 7 中 decode 跨块。
- **这是 prefix cache 唯一的淘汰路径**。系统没有主动淘汰机制，缓存只在块被重新分配时才失效。

---

## 三、串起来看

### 一条序列的一生

```
          add_request
              │
              ▼
   ┌──── waiting ◄──────────────────────────────┐
   │          │                                  │
   │  ① can_allocate ─ -1 ─► 继续等待（场景 3）   │
   │          │ ≥0                               │
   │  ② allocate（场景 1/2）                      │
   │          │                                  │
   │  [chunked prefill：多次 step，不再分配（场景 4）]
   │          ▼                                  │
   │       running                               │
   │          │  每个 decode step：               │
   │          │  ③ can_append / may_append       │
   │          │     （场景 6/7）                  │
   │          │     不够 → 抢占别人或自己 ──────────┤ ④ deallocate（场景 8 → 场景 10）
   │          │                                  │
   │          │  每一步结束：⑤ hash_blocks（场景 5）
   │          ▼                                  │
   │     EOS / max_tokens                        │
   │  ⑥ deallocate（场景 9）                      │
   │          ▼                                  │
   │       FINISHED                              │
   └─────────────────────────────────────────────┘
```

### 从函数反查场景

| 函数 | 在哪些场景中被调用 |
|---|---|
| `can_allocate` | 1、2、3、10（序列每次进入 prefill 且没有 block_table 时） |
| `allocate` | 1、2、10（`can_allocate` 不返回 -1 时） |
| `can_append` | 6、7、8（每个 decode step，每条序列一次；抢占后会重试） |
| `may_append` | 6、7（`can_append` 通过后） |
| `hash_blocks` | 4、5（每一步，每条被调度的序列一次） |
| `deallocate` | 8（被抢占）、9（正常结束） |
| `_allocate_block` | 1、2（未命中的部分）、7；内部会触发 11 |
| `_deallocate_block` | 8、9（引用计数降为 0 时） |
| `compute_hash` | `can_allocate`、`allocate`、`hash_blocks` 内部 |
| `Block.update` | 只在 `hash_blocks` 中调用 |
| `Block.reset` | 只在 `_allocate_block` 中调用 |

---

## 四、Chunked Prefill

### 1. 是什么

**Chunked prefill 就是把一个长 prompt 的 prefill 拆成若干段（chunk），分到多个 step 里依次计算。** 每一步只算一段，算完的 K/V 留在 KV cache 里，下一段接着往后算。最后一段算完时，才产出第一个生成的 token。

```
prompt 共 2500 个 token，每步的 token 预算为 1000

step 1：计算 [0, 1000)      → K/V 写入 cache，不产出 token
step 2：计算 [1000, 2000)   → 读取前 1000 个 token 的 K/V，写入新的 K/V，不产出 token
step 3：计算 [2000, 2500)   → 读取前 2000 个 token 的 K/V，写入新的 K/V，产出第一个 token
step 4 起：进入 decode
```

### 2. 为什么需要

**根本原因：每步的 token 数有上限。** 调度器每一步最多处理 `max_num_batched_tokens` 个 token（默认 16384，见 [config.py](../nanovllm/config.py)）。这个上限和显存规划直接相关：

- 启动时，[warmup_model](../nanovllm/engine/model_runner.py#L91-L101) 正是用 `max_num_batched_tokens` 个 token 跑一次 prefill，测出**激活值的峰值显存**；
- 然后从剩余显存中扣掉这个峰值，剩下的才分给 KV cache。

也就是说，系统只为"每步最多 `max_num_batched_tokens` 个 token"预留了激活值显存。如果一个 prompt 比这个上限还长：

- 一次性算完：激活值可能超出预留的显存，导致 OOM；
- 不允许调度：它会永远卡在 `waiting` 队头，后面的请求也都被堵住。

Chunked prefill 让这种 prompt 也能在预算内被处理：每步只算预算允许的那一段。

**在 vLLM 里还有另一个作用，但 nano-vllm 没有。** 在 vLLM / Sarathi-Serve 中，chunked prefill 还用来**和 decode 混在同一个 step 里执行**：每步先安排所有 decode token，再用剩余的预算处理一段 prefill。这样长 prompt 进来时，正在 decode 的请求不会被卡住。

nano-vllm **不把 prefill 和 decode 混在一起**，而且 prefill 优先。一个被切块的序列会一直留在 `waiting` 队头，所以它的所有 chunk 会**连续几步算完**，这期间 decode 是暂停的。因此在 nano-vllm 中，chunked prefill 的作用**只是"让超长 prompt 能放进每步的预算"**，没有改善延迟的效果。

### 3. 实现：调度、执行、后处理三处配合

#### (1) 调度：决定这一步算多少

[scheduler.py:30-53](../nanovllm/engine/scheduler.py#L30-L53) 的关键几行：

```python
remaining = self.max_num_batched_tokens - num_batched_tokens        # 本步还剩多少预算
...
num_tokens = seq.num_tokens - seq.num_cached_tokens                 # 这条序列还剩多少没算
if remaining < num_tokens and scheduled_seqs:                       # ① 只允许 batch 中第一条序列被切块
    break
if not seq.block_table:
    self.block_manager.allocate(seq, num_cached_blocks)             # ② 只在第一个 chunk 时分配块
seq.num_scheduled_tokens = min(num_tokens, remaining)               # ③ 本步最多算到预算用完
if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
    seq.status = RUNNING; self.waiting.popleft(); self.running.append(seq)   # ④ 最后一个 chunk 才移入 running
scheduled_seqs.append(seq)
```

- **① 只有第一条序列可以被切块**：如果 batch 里已经有别的序列，当前序列又放不下，就直接停止，留到下一步。所以只有当一条序列**独占一整步的预算仍然放不下**时，才会被切块。这样每一步最多只有一条序列处于"算了一半"的状态，逻辑最简单。
  - 默认配置下，`max_num_batched_tokens = 16384`，而 `max_model_len = 4096`，所以通常不会触发切块。把 `max_num_batched_tokens` 调小，或者处理很长的 prompt 时才会遇到。
- **② 块一次性分配**：第一个 chunk 时，`allocate` 就为整个 prompt 分配好所有块。后续 chunk 的 `block_table` 不为空，会跳过分配。
- **③ 用 `num_cached_tokens` 记录进度**：`num_cached_tokens` 表示已经写入 cache 的 token 数，本步从这里开始，计算 `num_scheduled_tokens` 个 token。
- **④ 没算完的序列留在 `waiting` 队头**：下一步调度时，它又会排在第一个，继续算下一段。

#### (2) 执行：Q 只包含本段，K/V 包含整个前缀

[prepare_prefill](../nanovllm/engine/model_runner.py#L129-L170)：

```python
start = seq.num_cached_tokens                 # 本段起点
end   = start + seq.num_scheduled_tokens      # 本段终点
input_ids = seq[start:end]                    # Q：只包含本段的 token
seqlen_q  = end - start
seqlen_k  = end                               # K：从 0 到 end，包含之前所有 chunk
```

从第二个 chunk 开始，`seqlen_k > seqlen_q`，于是会构造 `block_tables`。`Attention` 看到 `block_tables` 不为空，就改为从 `k_cache / v_cache` 中按块读取 K/V（[attention.py:65-66](../nanovllm/layers/attention.py#L65-L66)）。之前 chunk 的 K/V 已经在 cache 里，本段的 K/V 也在 attention 之前刚刚写进去，所以能读到完整的前缀。

以 step 2 为例：

```
Q：token[1000:2000]（1000 个）
K/V：token[0:2000]（2000 个，从 cache 中按 block_table 读取）
causal mask：右下对齐 → Q 中的第 j 个 token（实际位置为 1000+j）可以看到 K 中的 [0, 1000+j]
```

FlashAttention 在 `seqlen_q < seqlen_k` 时，causal mask 按**右下角对齐**，正好符合"Q 是 K 的最后一段"这种情况。

这套机制和 **prefix cache 命中**完全相同：两者都是"前缀已经在 cache 中，本步只计算后面的部分"。所以 nano-vllm 用同一段代码同时支持了这两种功能。

#### (3) 后处理：中间的 chunk 不产出 token

每个 chunk 的前向都会照常计算 logits、采样出一个 token。但对于中间的 chunk，这个 token 是没有意义的，因为 prompt 还没读完。[postprocess](../nanovllm/engine/scheduler.py#L81-L92) 会把它丢掉：

```python
self.block_manager.hash_blocks(seq)                       # 登记本段写满的块
seq.num_cached_tokens += seq.num_scheduled_tokens         # 推进进度
seq.num_scheduled_tokens = 0
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue                                              # prompt 还没算完 → 丢弃采样结果，不追加 token
seq.append_token(token_id)
```

另外，每个 chunk 写满的块都会通过 `hash_blocks` 及时登记。所以在这条长 prompt 还在 prefill 的过程中，后续前缀相同的请求就已经可以命中它前面的块了。

### 4. 小结

| 问题 | nano-vllm 的做法 |
|---|---|
| 什么时候切块 | 一条序列独占一整步的预算仍然放不下时（只有 batch 中第一条序列可以被切块） |
| 进度记在哪里 | `seq.num_cached_tokens` |
| KV 块什么时候分配 | 第一个 chunk 时为整个 prompt 一次性分配 |
| 后续 chunk 如何看到前面的 K/V | `seqlen_k > seqlen_q` → 构造 `block_tables` → 从分页 cache 读取 |
| 中间 chunk 的采样结果 | 在 `postprocess` 中丢弃 |
| 什么时候进入 decode | 最后一个 chunk 算完后，移入 `running` |
| 与 vLLM 的区别 | 不与 decode 混批，只用于解决超长 prompt 的显存预算问题，不改善延迟 |

---

## 五、分块计算会不会让显存溢出

正常情况下不会。推理时的显存可以分成两类：**每步临时用的激活值**，和**一直保留的 KV cache**。chunked prefill 只负责控制前一类；后一类由 BlockManager 保证。两类各有上限，所以不会溢出。

### 1. 两类显存

| | 激活值（activations） | KV cache |
|---|---|---|
| 是什么 | 前向过程中的中间张量：hidden states、QKV、MLP 的中间结果、logits 等 | 每个 token 每一层的 K/V |
| 生命周期 | **临时**：一步结束就释放 | **长期**：从写入开始，直到序列结束才释放 |
| 大小取决于 | **本步处理的 token 数** | **序列的总长度** |
| 由谁控制 | 每步 token 预算 `max_num_batched_tokens` + chunked prefill | 预分配的 KV 池 + BlockManager 的准入检查 |

启动时，显存就按这个思路分好了（[allocate_kv_cache](../nanovllm/engine/model_runner.py#L103-L121)）：

```
GPU 显存 × gpu_memory_utilization
 ├─ 模型权重
 ├─ 激活值峰值  ← warmup 时用 max_num_batched_tokens 个 token 实测得到，预留出来
 └─ KV cache 池 ← 剩下的全部切成 block，一次性分配好
```

### 2. 激活值：chunked prefill 保证不超过预留的峰值

一个 chunk 最多包含 `max_num_batched_tokens` 个 token，正好就是 warmup 时测峰值用的规模。所以**每一步的激活值都不会超过预留的量**。

后面的 chunk 要 attend 到很长的前缀，K 变长了，显存会不会跟着变大？不会，原因有三个：

1. **FlashAttention 不会生成 Q×K 的注意力矩阵**。普通 attention 要显式算出 `[len_q, len_k]` 的 scores 矩阵，K 越长，矩阵越大。FlashAttention 采用分块计算加在线 softmax，额外显存只和 Q 的长度有关，和 K 的长度无关。
2. **前缀的 K/V 直接从 cache 中原地读取**。传给 kernel 的是 `k_cache` 和 `block_table`（[attention.py:65-66](../nanovllm/layers/attention.py#L65-L66)），不会把前缀的 K/V 拷贝出一份连续的副本。
3. **LMHead 只算每条序列最后一个位置的 logits**（[embed_head.py](../nanovllm/layers/embed_head.py)）。输出是 `[序列数, vocab]`，而不是 `[token 数, vocab]`。

所以，一个 chunk 产生的激活值只和 **chunk 本身的长度**有关，和前缀有多长无关。

```
第 1 个 chunk：Q = 1000，K = 1000   → 激活值 ∝ 1000
第 3 个 chunk：Q = 1000，K = 3000   → 激活值仍然 ∝ 1000（K 在 cache 里原地读取）
```

### 3. KV cache：池子固定，按需在池内分配

KV cache 确实会随着 chunk 一段段增长，但它**不会去申请新的显存**：

1. **整个 KV 池在启动时就分配好了**，运行时不会调用 `cudaMalloc`。"分配一个块"只是从空闲队列里拿出一个整数。
2. **在第一个 chunk 之前，就检查整个 prompt 所需的块是否足够**。`can_allocate` 按 `seq.num_blocks`（整个 prompt 需要的块数）来检查，`allocate` 也一次性为整个 prompt 分配好所有块。如果块不够，这条序列就留在 `waiting` 中等待（场景 3），不会出现"算到一半发现没地方放"的情况。
3. decode 阶段 KV 继续增长，靠 `can_append` 加**抢占**来兜底（场景 7、8）：块不够时，先踢掉其他序列，腾出空间再继续。

所以 KV cache 的上限就是池的大小，序列只是在这个池子里分配和归还，不会超出。

### 4. 仍然可能出问题的边界情况

上面说的是正常路径。nano-vllm 为了保持代码精简，没有对输入长度做校验，下面几种情况会出错。它们都不是显存 OOM，而是报错：

| 情况 | 结果 |
|---|---|
| **单个 prompt 需要的块数超过整个 KV 池** | `can_allocate` 永远返回 -1，序列一直卡在 `waiting` 队头。等 `running` 中的序列都跑完，`schedule()` 什么都调度不到，[scheduler.py:71](../nanovllm/engine/scheduler.py#L71) 的 `assert scheduled_seqs` 报错 |
| **prompt 长度加上生成长度超过 `max_model_len`** | 代码不检查这一点。decode 走 CUDA Graph 时，`graph_vars["block_tables"]` 只有 `max_model_len / block_size` 列，序列的块数超过这个值，拷贝时会形状不匹配并报错（[model_runner.py:210](../nanovllm/engine/model_runner.py#L210)） |
| **warmup 之后显存环境发生变化** | 比如同一张卡上又启动了别的进程，或者显存碎片导致实际可用显存比测量时少。这种情况下仍可能 OOM，`gpu_memory_utilization` 默认设为 0.9 就是为了留出余量 |

vLLM 会在 `add_request` 时就拒绝超过 `max_model_len` 的请求，nano-vllm 省略了这一步。

### 5. 总结

- **Chunked prefill 管的是每步的计算量**：把一个超长 prompt 的激活值，切成每份都不超过预留峰值的若干段。
- **KV cache 由 BlockManager 管理**：池子是预先分配好的，第一个 chunk 之前就按整个 prompt 检查容量，decode 期间靠抢占兜底。
- FlashAttention 的显存与 K 长度无关，前缀 K/V 又是原地读取，所以 chunk 越往后，激活值也不会变大。

真正可能出问题的，是 prompt 超过 KV 池总容量或超过 `max_model_len` 这类没做校验的极端输入。

---

## 六、补充：slot_mapping 与 hash_blocks 的直观理解

下面用 [Sequence.md](Sequence.md) 中的例子（block_size = 4，prompt 共 6 个 token `t0..t5`，`block_table = [3, 7]`），直观说明 prefill 这一步中的两件事：KV 写到哪里，以及写满的块怎样登记以供复用。

### 1. slot_mapping：每个 token 的 KV 写到 cache 的哪个格子

可以把整个 KV cache 想象成一排编号连续的格子（slot），每 `block_size` 个格子组成一个块。在 block_size = 4 时：

```
块号:     0            1            2            3              ...    7
格子:  [0 1 2 3]   [4 5 6 7]   [8 9 10 11]  [12 13 14 15]  ...  [28 29 30 31]
```

所以，块 b 中第 k 个格子的编号是 `b × block_size + k`。

这条序列的 `block_table = [3, 7]`，意思是：

- 它的第 0 块（token 0~3）放在物理块 3；
- 它的第 1 块（token 4~7）放在物理块 7。

prefill 要写入 6 个 token 的 KV，每个 token 对应的格子是：

| token | 位置 | 所在逻辑块 | 物理块 | 块内偏移 | slot |
|---|---|---|---|---|---|
| t0 | 0 | 0 | 3 | 0 | 3×4+0 = **12** |
| t1 | 1 | 0 | 3 | 1 | **13** |
| t2 | 2 | 0 | 3 | 2 | **14** |
| t3 | 3 | 0 | 3 | 3 | **15** |
| t4 | 4 | 1 | 7 | 0 | 7×4+0 = **28** |
| t5 | 5 | 1 | 7 | 1 | **29** |

所以 `slot_mapping = [12, 13, 14, 15, 28, 29]`。每一层的 Attention 都按这个列表，把 6 个 token 的 K/V 写到对应的格子里（[store_kvcache](../nanovllm/layers/attention.py#L33)）。块 7 的格子 30、31 暂时空着，留给之后生成的 token。

### 2. hash_blocks：把写满的块登记下来，供以后复用

prefill 结束后：

- **物理块 3** 存满了 t0~t3 这 4 个 token 的 KV。这些内容**以后不会再变**；
- **物理块 7** 只存了 t4、t5，后面还会继续写入新 token，内容**还在变化**。

`hash_blocks` 做的事情就是：**给写满的块算一个"内容指纹"（hash），登记到一张表里**：

```
h = hash(t0, t1, t2, t3)
hash_to_block_id[h] = 3        ← "内容是 t0~t3 的 KV，存在物理块 3"
```

块 7 还没写满，不登记。

**登记的作用**：以后如果有一个新请求，它的 prompt 也是以 t0~t3 开头（比如相同的 system prompt），`can_allocate` 算出同样的 hash，查表发现块 3 里已经有现成的 KV，就直接让新请求的 `block_table` 也指向块 3，**跳过这 4 个 token 的计算**。这就是 prefix caching（对应第二节的场景 2 和场景 5）。

补充一点：下一块的 hash 会把前一块的 hash 也算进去，比如 `hash(h, t4..t7)`。这样 hash 相同，就意味着从开头到这一块的**整个前缀**都相同。因为同样的 token 接在不同的前缀后面，算出的 KV 是不一样的，不能复用。

更详细的分析见 [PagedAttention.md](PagedAttention.md) 第四节。
