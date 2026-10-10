# Sequence：一条请求在引擎内部的抽象

[sequence.py](../nanovllm/engine/sequence.py) 只有 83 行。用户提交的是一个 prompt 字符串，进入引擎之后，它就变成一个 `Sequence` 对象，并且一直以这个对象的形式存在，直到生成结束。

本文先按用途把字段分组，再用一个例子跟踪每个字段在请求生命周期中的变化，最后说明跨进程序列化的设计，以及代码中的一个潜在问题。

相关文档：[Scheduler.md](Scheduler.md)、[block.md](block.md)、[ModelRunner.md](ModelRunner.md)。

---

## 一、字段分组

`Sequence` 同时回答了五个问题：这条请求是谁、内容是什么、进行到哪一步、KV 存在哪里、怎么采样。

| 组 | 字段 / 属性 | 作用 | 谁读写 |
|---|---|---|---|
| **身份** | `seq_id` | 全局自增编号（类属性 `counter`），`generate()` 最后按它恢复提交顺序 | LLMEngine |
| **内容** | `token_ids` | prompt 加上已生成的全部 token | 全部组件 |
| | `num_tokens` | token 总数。`len(seq)` 返回的就是它 | 全部组件 |
| | `num_prompt_tokens` | prompt 长度，**固定不变**，用来区分 prompt 和生成部分 | 派生属性 |
| | `last_token` | 最后一个 token，decode 时的模型输入 | ModelRunner |
| **生命周期** | `status` | WAITING / RUNNING / FINISHED | Scheduler |
| | `is_prefill` | 下一次被执行时是 prefill 还是 decode，决定跨进程时传输多少数据 | Scheduler 写，`__getstate__` 读 |
| **进度** | `num_cached_tokens` | **KV 已经写入 cache 的 token 数** | Scheduler、BlockManager、ModelRunner |
| | `num_scheduled_tokens` | 本步要计算的 token 数，每步由 Scheduler 设置，step 结束后清零 | Scheduler 写，ModelRunner 读 |
| **KV 映射** | `block_table` | 逻辑块 → 物理块 | BlockManager 写，ModelRunner 读 |
| | `num_blocks` | 当前 token 数需要几个块：`ceil(num_tokens / block_size)` | BlockManager |
| | `last_block_num_tokens` | 最后一块中有几个 token，decode 时用来计算写入位置 | ModelRunner |
| | `block(i)` | 第 i 块的 token 列表，用于计算 prefix cache 的 hash | BlockManager |
| **采样参数** | `temperature`、`max_tokens`、`ignore_eos` | 从 `SamplingParams` 拷贝过来，每条请求可以不同 | Sampler、Scheduler |
| **派生信息** | `num_completion_tokens`、`prompt_token_ids`、`completion_token_ids`、`is_finished` | 由上面的字段计算得到 | Scheduler、LLMEngine |

整个类只有 `append_token` 这一个"写内容"的方法，其余字段要么在构造时确定，要么由 Scheduler 和 BlockManager 直接赋值。

---

## 二、用一个例子跟踪字段变化

为了方便演示，设 `block_size = 4`（实际默认是 256）。假设 prompt 编码后有 6 个 token `[t0..t5]`，参数为 `SamplingParams(temperature=0.6, max_tokens=3)`。

### T0：`add_request` 创建对象

```python
seq = Sequence([t0, t1, t2, t3, t4, t5], sp)
```

```
seq_id = 0                    status = WAITING             is_prefill = True
token_ids = [t0..t5]          num_tokens = 6               num_prompt_tokens = 6
last_token = t5               num_cached_tokens = 0        num_scheduled_tokens = 0
block_table = []              num_blocks = 2               last_block_num_tokens = 2
temperature = 0.6             max_tokens = 3               ignore_eos = False
```

这时它只是一个"待处理的请求"，还没有分配任何 KV 块。

### T1：第 1 步，prefill

**schedule**：

- `block_table` 为空 → `can_allocate` 返回 0（没有命中前缀）→ `allocate` 分配两个块，假设是 3 号和 7 号：`block_table = [3, 7]`；
- 设置 `num_scheduled_tokens = 6 - 0 = 6`；
- `0 + 6 == num_tokens`，说明 prompt 本步可以全部算完 → `status = RUNNING`，移入 `running`。

**run**（ModelRunner 读取的字段）：

```
区间 [num_cached_tokens, num_cached_tokens + num_scheduled_tokens) = [0, 6)
input_ids = token_ids[0:6]，positions = 0..5
slot_mapping = 块 3 的槽位 12~15，加上块 7 的槽位 28、29
```

**postprocess**：

```
hash_blocks：本步写满了第 0 块（物理块 3）→ 登记 hash
num_cached_tokens = 0 + 6 = 6，num_scheduled_tokens = 0
append_token(g0) → token_ids = [t0..t5, g0]，num_tokens = 7，last_token = g0
num_completion_tokens = 7 - 6 = 1 < 3 → 继续
```

> `slot_mapping` 和 `hash_blocks` 的直观解释见 [block.md 第六节](block.md#六补充slot_mapping-与-hash_blocks-的直观理解)。

### T2：第 2 步，decode

**schedule**：

- `len(seq) = 7`，`7 % 4 = 3`，不需要新块；
- 设置 `num_scheduled_tokens = 1`，`is_prefill = False`。

**run**：

```
input_ids = [last_token] = [g0]，position = len - 1 = 6
last_block_num_tokens = 7 - (2-1) × 4 = 3
slot = block_table[-1] × 4 + last_block_num_tokens - 1 = 7 × 4 + 2 = 30   → 块 7 的第 2 个位置，对应下标 6 ✓
context_lens = 7
```

**postprocess**：`num_cached_tokens = 7`，追加 g1 → `num_tokens = 8`

### T3：第 3 步，decode，然后结束

**schedule**：`8 % 4 = 0`，不需要新块。

**run**：position 7，slot = 7 × 4 + 3 = 31，块 7 被写满。

**postprocess**：

```
hash_blocks：第 1 块（物理块 7）写满 → 登记 hash
num_cached_tokens = 8
append_token(g2) → num_tokens = 9，num_completion_tokens = 3 == max_tokens
→ status = FINISHED → deallocate：block_table = []，num_cached_tokens = 0
→ 从 running 中移除；LLMEngine 返回 (seq_id = 0, completion_token_ids = [g0, g1, g2])
```

### 字段变化一览

| 时刻 | status | is_prefill | num_tokens | num_cached_tokens | num_scheduled_tokens | block_table | last_token |
|---|---|---|---|---|---|---|---|
| T0 创建 | WAITING | True | 6 | 0 | 0 | `[]` | t5 |
| T1 schedule 后 | RUNNING | True | 6 | 0 | 6 | `[3, 7]` | t5 |
| T1 postprocess 后 | RUNNING | True | 7 | 6 | 0 | `[3, 7]` | g0 |
| T2 schedule 后 | RUNNING | False | 7 | 6 | 1 | `[3, 7]` | g0 |
| T2 postprocess 后 | RUNNING | False | 8 | 7 | 0 | `[3, 7]` | g1 |
| T3 postprocess 后 | FINISHED | False | 9 | 0 | 0 | `[]` | g2 |

### 几个值得注意的点

- **`num_tokens` 和 `num_cached_tokens` 通常差 1**：decode 阶段，刚生成的 token 已经追加到 `token_ids` 中，但它的 KV 要到下一步作为输入时才会写入 cache。结束时，g2 的 KV 从来没有被计算过，因为不需要。
- **`num_blocks` 可能比 `len(block_table)` 多 1**：T3 结束前 `num_tokens = 9`，`num_blocks = 3`，但 `block_table` 只有 2 块。`num_blocks` 是"全部 token 需要几块"，而 block_table 只在真正要写入 KV 时才扩展（`may_append` 判断 `len % block_size == 1` 就是为了这个时机）。
- **`num_prompt_tokens` 永远不变**：它是 prompt 和生成部分的分界线。`completion_token_ids` 就是 `token_ids[num_prompt_tokens:]`。
- **`is_prefill` 和 `status` 不是一回事**：prefill 完成后，`status` 立即变为 RUNNING，但 `is_prefill` 要等到第一次被调度执行 decode 时，才由 Scheduler 改为 False。

---

## 三、其他场景下字段的变化

| 场景 | 字段变化 |
|---|---|
| **chunked prefill**（prompt 超出 token 预算） | 第一步：`num_scheduled_tokens < num_tokens - num_cached_tokens`，`status` 保持 WAITING，留在 waiting 队头；每步结束 `num_cached_tokens` 前进一段，中间段的采样结果被丢弃；最后一段完成时才变为 RUNNING |
| **prefix cache 命中** | `allocate` 直接设置 `num_cached_tokens = 命中块数 × block_size`，prefill 从这里开始计算 |
| **被抢占** | `status = WAITING`，`is_prefill = True`，`block_table = []`，`num_cached_tokens = 0`；**`token_ids` 保留已生成的 token**。重新调度时，prompt 加上已生成部分会一起作为 prefill 输入。`num_prompt_tokens` 不变，所以生成计数不受影响 |

从这三个场景可以看出，Scheduler 不需要额外的状态来区分"新请求""续算的 chunk""被抢占后恢复"。**这些情况都只用 `num_cached_tokens` 和 `num_tokens` 之间的差值来表达"还剩多少没算"**，ModelRunner 也只按 `[num_cached_tokens, num_cached_tokens + num_scheduled_tokens)` 这个区间取数据。这就是这个抽象的关键。

---

## 四、`__getstate__`：跨进程时只传必要的数据

张量并行时，rank 0 每步都要把 `seqs` 用 pickle 序列化，经共享内存传给其他 rank（见 [ModelRunner.md](ModelRunner.md)）。[sequence.py:72-83](../nanovllm/engine/sequence.py#L72-L83)：

```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)

def __setstate__(self, state):
    self.num_tokens, ..., self.block_table, last_state = state
    if isinstance(last_state, list):        # prefill：传的是完整的 token_ids
        self.token_ids = last_state
        self.last_token = self.token_ids[-1]
    else:                                   # decode：只传最后一个 token
        self.token_ids = []
        self.last_token = last_state
```

**传了什么**：只有 ModelRunner 的 `prepare_prefill` / `prepare_decode` 需要用到的字段。

**没传什么**：

- `seq_id`、`status`：执行层不需要；
- `temperature`：只有 rank 0 采样，而 rank 0 用的是本地的原始对象；
- `max_tokens`、`ignore_eos`：停止条件由 Scheduler 判断，只在 rank 0 上进行。

**为什么 decode 只传 `last_token`**：decode 时模型输入只需要最后一个 token，历史信息已经在 KV cache 里，通过 `block_table` 读取。一条已经生成了 2000 个 token 的序列，decode 时传 1 个整数就够了，而不是 2000 个。

**这也解释了为什么要单独保存 `num_tokens` 和 `last_token`**：子进程反序列化后，decode 阶段的 `token_ids` 是空列表，但：

- `len(seq)` 返回的是 `num_tokens`，不是 `len(token_ids)`，所以仍然正确；
- `last_token` 单独保存，不需要从 `token_ids[-1]` 中取。

`prepare_decode` 只用到 `last_token`、`len(seq)`、`block_table`、`last_block_num_tokens`，这些在子进程中都是完整的。如果 `__len__` 返回的是 `len(self.token_ids)`，这套精简传输就无法成立。

---

## 五、潜在问题：子进程中的 `block_size`

`block_size` 是**类属性**，默认值为 256（[sequence.py:15](../nanovllm/engine/sequence.py#L15)）。[llm_engine.py:21](../nanovllm/engine/llm_engine.py#L21) 在主进程中把它改成配置值：`Sequence.block_size = config.kvcache_block_size`。

但张量并行的子进程是用 `spawn` 方式启动的，会重新 import 模块，所以子进程中的 `Sequence.block_size` 仍然是默认的 256。而子进程的 `prepare_decode` 会用 `seq.last_block_num_tokens` 计算写入位置，这个属性依赖 `block_size`。

因此，按代码推断，**当 `tensor_parallel_size > 1` 并且 `kvcache_block_size` 不是 256（比如 512）时，rank > 0 计算出的 decode 写入位置会出错**。

举例：`num_tokens = 300`，block_size = 512。

- 主进程：`num_blocks = 1`，`last_block_num_tokens = 300`；
- 子进程：`num_blocks = 2`，`last_block_num_tokens = 44`，写入位置错误。

在默认配置（256）下不会触发。修复方法很简单：在 `ModelRunner.__init__` 中也设置一次 `Sequence.block_size = config.kvcache_block_size`。

> 这一点是阅读代码得出的推断，尚未在多卡环境中实际运行验证。
