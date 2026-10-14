# Scheduler 设计

[scheduler.py](../../nanovllm/engine/scheduler.py) 只有 92 行，但做了一连串设计取舍。本文先说它在系统中的位置，再按"设计一个调度器必须回答的问题"逐条拆解：每个问题有哪些做法、nano-vllm-plus 选了哪种、为什么这样选、代价是什么。

块分配、chunked prefill 的实现细节见 [block.md](block.md) 和 [PagedAttention.md](PagedAttention.md)，这里只从调度策略的角度提及。整体架构见 [overview.md](overview.md)。

---

## 一、定位：只做决策，不碰张量

```
LLMEngine.step()
  ├─ Scheduler.schedule()      → (list[Sequence], is_prefill)    决定"这一步跑谁、各跑多少"
  ├─ ModelRunner.run(...)      → list[int]                       执行
  └─ Scheduler.postprocess()                                     推进状态、回收资源
```

| 持有的东西 | 说明 |
|---|---|
| `waiting: deque[Sequence]` | 还没完成 prefill 的序列：新请求、被抢占的请求、prefill 只做了一部分的请求 |
| `running: deque[Sequence]` | 已完成 prefill、正在 decode 的序列 |
| `block_manager` | KV 块的唯一管理者，只有 Scheduler 调用它 |
| `max_num_seqs`、`max_num_batched_tokens`、`eos` | 预算和停止条件 |

**边界**：Scheduler 只读写 `Sequence` 上的整数字段（`num_cached_tokens`、`num_scheduled_tokens`、`block_table`、`status`），不创建任何张量，也不知道模型是什么。它和执行层之间只有一个约定：`num_scheduled_tokens` 表示本步要算多少 token，`is_prefill` 表示本步是哪种类型。

---

## 二、设计一个调度器要回答的问题

### 问题 1：调度的粒度是什么？

| 方案 | 说明 |
|---|---|
| 请求级（static batching） | 一批请求一起开始、一起结束 |
| **迭代级（iteration-level）** | 每次前向之前都重新决定 batch 的成员 |

**nano-vllm-plus 的选择**：迭代级。`LLMEngine.step()` 每调用一次，就执行一次 `schedule()`。这是 continuous batching 的基础，原因见 [ContinuousBatching.md](ContinuousBatching.md)。

### 问题 2：序列的状态怎么建模？

**nano-vllm-plus 的选择**：三种状态，其中两种用队列表示。

```
          add()                  prefill 全部算完                 EOS / max_tokens
  新请求 ───────► waiting ──────────────────────► running ─────────────────────► FINISHED（不保存，直接丢弃）
                    ▲                                │
                    └────────── preempt ─────────────┘
```

- **FINISHED 的序列不放在任何队列里**：`postprocess` 会把它从 `running` 中移除，然后由 `LLMEngine.step()` 返回给调用方。
- **用 `deque` 实现**：队头取出、队头插回（被抢占的序列）、队尾抢占（选择被抢占者），这些都是 O(1) 操作。
- **`waiting` 里不只有新请求**：被抢占的序列、chunked prefill 只做了一部分的序列也在里面。它们共同的特点是"还需要 prefill"。所以状态的含义不是"新"还是"旧"，而是"下一步需要做 prefill 还是 decode"。

### 问题 3：prefill 和 decode 要不要放进同一个 batch？

这是调度器最核心的一个取舍。

| 方案 | 优点 | 缺点 |
|---|---|---|
| **分开执行，prefill 优先**（nano-vllm-plus、早期 vLLM） | 每步只有一种计算模式，逻辑简单；decode 步可以完整使用 CUDA Graph | 新请求到来时 decode 要暂停，生成 token 的间隔不均匀 |
| 分开执行，decode 优先 | decode 延迟平稳 | 新请求的首 token 延迟（TTFT）变差；batch 较小，吞吐下降 |
| 混合执行（vLLM v1、Sarathi-Serve） | 吞吐和延迟兼顾 | attention 要同时处理两种情况；混合 batch 难以使用 CUDA Graph；调度逻辑复杂 |

**nano-vllm-plus 的选择**：分开执行，prefill 优先（[scheduler.py:29-55](../../nanovllm/engine/scheduler.py#L29-L55)）：

```python
# prefill
while self.waiting and ...:
    ...
if scheduled_seqs:
    return scheduled_seqs, True      # 只要调度到一条 prefill，本步就只做 prefill
# decode
while self.running and ...:
    ...
```

**这样选的理由**：

1. **符合离线场景的目标**：nano-vllm-plus 的 `generate()` 一次性提交所有请求，关注的是**总吞吐**，而不是单个请求的延迟抖动。prefill 优先能尽快把 `running` 填满，让 decode 的 batch 尽可能大。
2. **执行层可以保持简单**：`is_prefill` 只是一个 bool，ModelRunner 根据它选择 `prepare_prefill` 或 `prepare_decode`，Attention 根据它选择 `flash_attn_varlen_func` 或 `flash_attn_with_kvcache`。decode 步的形状规整，可以直接 replay CUDA Graph。

**不会出现死等**：如果 `waiting` 不为空，但队头的序列放不下（块不够），这一步就调度不到任何 prefill，会自动转去执行 decode。等 decode 中有序列结束、释放了块，下一步再试。

### 问题 4：每一步受哪些资源约束？

调度器要同时满足三种资源的约束：

| 约束 | 限制的是什么 | 由什么决定 | 在哪一阶段检查 |
|---|---|---|---|
| `max_num_seqs`（默认 512） | 一步最多几条序列 | batch 规模上限；CUDA Graph 最大只录制到 512 档 | prefill 和 decode 都检查 |
| `max_num_batched_tokens`（默认 16384） | 一步最多几个 token | **激活值显存**：warmup 正是按这个规模测出的峰值 | 只在 prefill 检查 |
| KV 空闲块数 | 能否放下 KV | **KV cache 显存** | prefill 用 `can_allocate`，decode 用 `can_append` |

decode 不检查 token 预算：每条序列只算 1 个 token，所以总 token 数等于序列数，不超过 512，远小于 16384。

这三项约束分别对应三种不同的资源：batch 规模、激活值显存、KV 显存。warmup 测量和 KV 池的分配，都是按这些预算来做的，所以**只要调度器守住这三条，执行层就不会 OOM**（见 [block.md 第五节](block.md#五分块计算会不会让显存溢出)）。

### 问题 5：按什么顺序接纳新请求？

| 方案 | 说明 |
|---|---|
| **严格 FCFS** | 只看队头，队头放不下就停止 |
| FCFS + 跳过 | 队头放不下时，跳过它，尝试后面较小的请求 |
| 优先级 / SJF | 按优先级或预估长度排序 |

**nano-vllm-plus 的选择**：严格 FCFS。prefill 循环始终只看 `waiting[0]`，有三种情况会直接 `break`：

```python
if remaining == 0: break                                  # token 预算用完
if num_cached_blocks == -1: break                         # KV 块不够
if remaining < num_tokens and scheduled_seqs: break       # 放不下，而且不是本步的第一条
```

**理由**：

- **公平，不会饿死长请求**：如果允许跳过，长 prompt 可能一直被短请求插队。
- **和抢占配合**：被抢占的序列会插回 `waiting` **队头**。严格 FCFS 保证它下一次最先被重新调度，不会因为被抢占而排到最后。

**代价**：队头阻塞。一个放不下的大请求，会挡住后面本来可以放下的小请求。

### 问题 6：太长的 prompt 怎么办？

**nano-vllm-plus 的选择**：chunked prefill，并且只允许 batch 中的第一条序列被切块（[scheduler.py:42](../../nanovllm/engine/scheduler.py#L42)）。

- 只有当一条序列**独占整个 token 预算仍然放不下**时才会切块。这样每一步最多只有一条"算了一半"的序列，状态最容易管理。
- 没算完的序列不出队，下一步仍然排在队头，继续算下一段。这也是严格 FCFS 带来的结果。
- 它解决的是"放进预算"的问题，不改善延迟。详见 [block.md 第四节](block.md#四chunked-prefill)。

### 问题 7：显存不够了怎么办？

这个问题要分 prefill 和 decode 两个阶段来看，两者的处理方式不对称：

| 阶段 | 显存不够时 | 理由 |
|---|---|---|
| prefill（接纳新请求） | **等待**：不接纳，也不抢占别人 | 新请求还没开始算，等待的代价最小 |
| decode（已在运行的序列需要新块） | **抢占**：踢掉其他序列 | 不能让正在运行的序列卡住 |

**抢占时选谁？** [scheduler.py:58-65](../../nanovllm/engine/scheduler.py#L58-L65)：

```python
seq = self.running.popleft()                 # 按顺序处理 running 中的序列
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())     # 抢占 running 队尾，也就是最晚进入的序列
    else:
        self.preempt(seq)                    # 没有别人可抢了，就抢占自己
        break
else:                                        # while 正常结束（没有 break）才执行这里
    ... 调度 seq
```

- **选队尾（LIFO）**：队尾是最晚进入 running 的序列，通常已生成的 token 最少，重新计算的代价最低。同时也符合 FCFS 的公平原则：先来的请求优先保住。
- **抢占自己**：如果其他序列都被抢光了还是不够，就把自己也放回 waiting。这时 `while ... else` 的 `else` 分支不会执行，这条序列不进入本步的 batch。
- **抢占方式选择重算（recompute），而不是换出（swap）**：[preempt](../../nanovllm/engine/scheduler.py#L75-L79) 直接释放所有块，把序列插回 `waiting` 队头，并标记 `is_prefill = True`。重新调度时，**prompt 加上已生成的 token 会一起作为 prefill 输入**重新计算。因为 `num_prompt_tokens` 没有变，已生成 token 的计数也不受影响。不做 swap 可以省掉 CPU 内存管理和数据搬运的逻辑；而且被释放的块进入 FREE_CACHED 状态，重新调度时很可能命中 prefix cache，重算的代价往往比看上去小。

### 问题 8：running 队列的顺序怎么维护？

decode 分支用 `popleft` 依次取出序列，最后用 `extendleft(reversed(scheduled_seqs))` 把它们放回去（[scheduler.py:72](../../nanovllm/engine/scheduler.py#L72)）：

```
调度前：running = [A, B, C, D, E]，max_num_seqs = 3
popleft 取出 A、B、C 并调度 → running = [D, E]
extendleft(reversed([A,B,C])) → running = [A, B, C, D, E]   ← 原顺序不变
```

如果中途发生抢占，被抢占的是队尾的 E、D，它们直接离开 running，其余序列的相对顺序不变。

**目的**：让 running **始终按进入时间排序**。这样"队尾就是最晚进入的序列"这个前提一直成立，抢占时选择受害者的规则才正确。

**副作用**：当 running 中的序列数超过 `max_num_seqs` 时，排在后面的序列要等前面的序列结束后才能被调度。这和 FCFS 的思路一致。

### 问题 9：每一步结束后怎么推进状态？

[postprocess](../../nanovllm/engine/scheduler.py#L81-L92) 对本步的每条序列依次执行：

```
① hash_blocks             登记本步写满的块（prefix cache）
② num_cached_tokens += …  推进进度
③ prefill 还没算完？       → 丢弃本步的采样结果（chunked prefill 的中间段）
④ append_token            追加新 token
⑤ EOS 或达到 max_tokens？  → FINISHED，deallocate 释放块，从 running 中移除
```

**设计要点**：

- **"结束即释放"**：块在序列结束的同一步就被释放，下一步就能用于接纳新请求。continuous batching 的吞吐优势正是来自这里。
- **`num_cached_tokens` 是唯一的进度记录**：chunked prefill、prefix cache 命中、decode 都用它来表示"已经写入 cache 的 token 数"，调度和执行双方都以它为准。
- **停止条件比较简单**：只判断 EOS（可以用 `ignore_eos` 关闭）和 `max_tokens`，不支持 stop strings 等更复杂的条件。

---

## 三、`schedule()` 的完整决策流程

```
schedule()
│
├─ 【prefill 阶段】只看 waiting 队头，满足严格 FCFS
│   while waiting 不为空 且 已调度序列数 < max_num_seqs：
│     ├─ token 预算用完了？                         → 停止
│     ├─ 第一次调度（block_table 为空）：
│     │    can_allocate → -1？                      → 停止（等待，不抢占）
│     │    剩余要算的 = 总 token 数 − 命中前缀
│     │  否则（chunked prefill 的后续段）：
│     │    剩余要算的 = 总 token 数 − num_cached_tokens
│     ├─ 剩余预算放不下，而且不是本步第一条？           → 停止
│     ├─ 第一次调度 → allocate
│     ├─ num_scheduled_tokens = min(剩余要算的, 剩余预算)
│     └─ prompt 全部算完？ → 出队，转入 running；否则留在 waiting 队头
│   调度到了任何序列 → return (seqs, is_prefill=True)
│
└─ 【decode 阶段】按 running 的顺序处理
    while running 不为空 且 已调度序列数 < max_num_seqs：
      ├─ popleft 取出 seq
      ├─ 块不够 → 从队尾抢占别人；没有别人了 → 抢占自己
      └─ 块够 → num_scheduled_tokens = 1，may_append
    把已调度的序列按原顺序放回 running 队头
    return (seqs, is_prefill=False)
```

---

## 四、设计取舍总结

| 设计问题 | nano-vllm-plus 的选择 | 带来的好处 | 付出的代价 |
|---|---|---|---|
| 调度粒度 | 每步重新调度（迭代级） | continuous batching | 每步都有一些 CPU 开销 |
| prefill 和 decode | 分开执行，prefill 优先 | 执行层简单，decode 可以使用 CUDA Graph，离线吞吐高 | 新请求到来时 decode 暂停一步，token 间隔有抖动 |
| 资源约束 | 序列数、token 数、KV 块数三项预算 | 和启动时的显存规划一一对应，不会 OOM | 预算是静态配置，不会自适应调整 |
| 接纳顺序 | 严格 FCFS | 公平；被抢占的序列优先恢复 | 队头阻塞 |
| 长 prompt | chunked prefill，只允许第一条序列被切块 | 超长 prompt 也能放进预算 | 不改善延迟 |
| 显存不足 | prefill 等待；decode 抢占队尾，采用重算 | 实现简单；可以借助 prefix cache 减少重算 | 被抢占的序列要重新做 prefill |
| running 顺序 | 始终保持进入时间顺序 | 抢占选择受害者的规则一直正确 | 超出 `max_num_seqs` 的序列要排队 |
| 资源回收 | 序列结束的同一步就释放 | 下一步就能接纳新请求 | 无 |

### nano-vllm-plus 没有考虑的部分

这些在 vLLM 中都有，nano-vllm-plus 为了保持简洁而省略了：

- 请求优先级、SLO 感知调度、公平性策略（目前只有 FCFS）；
- prefill 和 decode 混合执行；
- swap 到 CPU；
- 在 `add_request` 时校验长度（`max_model_len`、KV 池总容量），见 [block.md 第五节](block.md#五分块计算会不会让显存溢出)；
- 取消请求、流式输出、在线服务；
- 调度和执行的异步重叠（下一步的调度与本步的 GPU 计算并行）；
- 一个请求生成多个候选（n > 1）、beam search。
