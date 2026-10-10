# PagedAttention 原理与 nano-vllm 的实现

本文先讲 PagedAttention 要解决的问题和核心思想，再按层拆解 nano-vllm 的实现。重点是逻辑层的 `Block` 与 `BlockManager`：逐个方法分析它们做了什么、为什么这么写，并用一个完整的状态追踪例子串起来。

相关文档：[overview.md](overview.md)、[ModelRunner.md](ModelRunner.md)、[ContinuousBatching.md](ContinuousBatching.md)。

---

## 一、PagedAttention 要解决什么问题

### 1. KV cache 有多大

每个 token 在每一层都要缓存一份 K 和一份 V。以 Qwen3-0.6B 为例（28 层，8 个 KV 头，head_dim 为 128，bf16）：

```
每个 token 的 KV = 2(K、V) × 28 层 × 8 头 × 128 × 2 字节 ≈ 112 KB
一条 4096 token 的序列 ≈ 448 MB
```

模型权重只有约 1.2 GB，而几十条序列的 KV cache 就能占满显存。**能同时跑多少条序列，取决于 KV cache 能放下多少**，KV cache 的管理效率直接决定了吞吐。

### 2. 连续分配的三种浪费

传统做法是给每条序列**预留一段连续的、按最大长度计算的空间**。vLLM 论文把由此产生的浪费分成三类：

```
序列 A 的预留区域（max_len = 8）
┌──┬──┬──┬──┬──┬──┬──┬──┐
│p │p │p │g │  │  │  │  │   p = prompt   g = 已生成
└──┴──┴──┴──┴──┴──┴──┴──┘
            ↑ reserved：将来"可能"用到，但现在别人也不能用
                        ↑ internal fragmentation：序列实际在第 5 个 token 结束，后面的永远用不上

    ┌────A────┐  空 3  ┌────B────┐ 空 2 ┌──C──┐
    ← external fragmentation：总空闲有 5，但没有一段连续空间放得下长度为 4 的新请求
```

vLLM 论文统计，当时的系统里 KV cache 只有 **20%–38%** 真正存放了有用的数据。

### 3. 核心思想：借鉴操作系统的虚拟内存

| 操作系统 | PagedAttention |
|---|---|
| 进程 | 序列（Sequence） |
| 虚拟页 | 逻辑块：序列的第 0、1、2…… 块 |
| 物理页框 | 物理块：KV cache 大张量中的第 k 块 |
| 页表 | block table |
| 缺页时分配 | 写满当前块时才分配新块 |
| 共享页（如 fork 后共享） | 多条序列共享相同前缀的块（引用计数） |

做法如下：

- 把 KV cache 切成**固定大小的块**，每块存 `block_size` 个 token；
- 每条序列只记录"我的第 i 个逻辑块在第几号物理块"，**物理块不需要连续**；
- 按需一块一块地分配，序列结束时整体归还。

效果是：

- **external fragmentation 消失**：所有块大小相同，任何空闲块都能用；
- **reserved 浪费消失**：用到才分配；
- **internal fragmentation 减少**：每条序列最多浪费最后一块里没填满的部分；
- **天然支持共享**：两条序列的 block table 指向同一个物理块即可。

### 4. 代价：attention kernel 要能"跳着读"

普通 attention 假设一条序列的 K/V 在显存中是连续的。分页之后，K/V 散落在不同的物理块里，kernel 必须先查 block table 才能读到数据。vLLM 最初为此专门写了 paged attention kernel；现在 FlashAttention 也通过 `block_table` 参数原生支持分页 KV。nano-vllm 直接使用后者。

---

## 二、nano-vllm 的三层实现

```
┌─ 逻辑层（CPU，scheduler 所在进程）─────────────────────────────┐
│  BlockManager：哪些块空闲、谁在用、引用计数、前缀 hash           │
│  Sequence.block_table：[7, 2, 9]                                │
├─ 地址翻译层（ModelRunner）──────────────────────────────────────┤
│  写：slot_mapping    每个新 token → 物理槽位号                   │
│  读：block_tables    每条序列 → 物理块号列表                     │
├─ 物理层（GPU）─────────────────────────────────────────────────┤
│  kv_cache 大张量 + store_kvcache（写）+ FlashAttention（读）     │
└────────────────────────────────────────────────────────────────┘
```

- **逻辑层**只处理整数，不碰任何张量；
- **地址翻译层**每步把整数翻译成 GPU 张量；
- **物理层**按翻译好的地址读写显存。

---

## 三、物理层：一块大张量

[model_runner.py `allocate_kv_cache`](../../nanovllm/engine/model_runner.py#L103-L121)：

```python
self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
# 第 i 层的 Attention：
module.k_cache = self.kv_cache[0, i]   # [num_blocks, block_size, kv_heads, head_dim]
module.v_cache = self.kv_cache[1, i]
```

- **启动时一次性分配**，运行过程中不再调用 `cudaMalloc`。"分配一个块"只是在 CPU 上从空闲队列里取出一个整数。
- **同一个 block id 在所有层通用**：物理块 7 指的是每一层 `k_cache[7]` 和 `v_cache[7]` 这一组位置。所以逻辑层只需要管理一套 block id，不用每层各管一套。
- 块的数量由显存测量结果决定，见 [ModelRunner.md](ModelRunner.md#-kv-cache-容量规划)。

**块大小为什么是 256**：[config.py](../../nanovllm/config.py#L22) 中有 `assert kvcache_block_size % 256 == 0`。这是 FlashAttention 分页 KV 接口的要求：使用 `block_table` 时，page block size 必须是 256 的倍数。

---

## 四、逻辑层：Block 与 BlockManager

代码在 [block_manager.py](../../nanovllm/engine/block_manager.py)，共 120 行。下面先看数据结构和不变式，再看块的状态机，最后逐个方法分析。

### 1. 数据结构

```python
class Block:                        # 一个物理块的元信息（显存里的数据不在这里）
    block_id: int                   # 物理块号，即 kv_cache 第 3 维的下标
    ref_count: int                  # 有几条序列的 block_table 引用了它
    hash: int                       # 内容的链式 hash；-1 表示"未登记"
    token_ids: list[int]            # 块内容的副本，用于校验 hash 碰撞

class BlockManager:
    blocks: list[Block]             # 全部物理块，下标就是 block_id
    free_block_ids: deque[int]      # 空闲块，FIFO：从队头取，往队尾放
    used_block_ids: set[int]        # 正在被引用的块
    hash_to_block_id: dict[int,int] # 前缀 hash → 物理块号（prefix cache 的索引）
```

还有 `Sequence` 中与之相关的字段（[sequence.py](../../nanovllm/engine/sequence.py)）：

| 字段 | 含义 |
|---|---|
| `block_table: list[int]` | 第 i 个元素是逻辑块 i 对应的物理块号 |
| `num_cached_tokens` | **KV 已经写入 cache 的 token 数**，下标 `< num_cached_tokens` 的 token 都已在 cache 中 |
| `num_scheduled_tokens` | 本步要计算（并写入 cache）的 token 数 |
| `num_blocks` | `ceil(num_tokens / block_size)`，当前 token 数需要的块数 |
| `block(i)` | 第 i 个逻辑块包含的 token id 列表 |

两个容易混淆的点：

- **`Block` 里没有 KV 数据**。KV 数据在 GPU 的 `kv_cache[:, :, block_id]` 里。`Block` 只是 CPU 端的"户口本"。
- **`Block.token_ids` 只在登记 hash 时才写入**（`update`），平时是空列表。它只用于 prefix cache 的校验，和 KV 的读写无关。

### 2. 不变式

BlockManager 的所有方法都在维护下面几条性质。理解了它们，代码就很好读：

1. **每个块要么在 `free_block_ids` 中，要么在 `used_block_ids` 中，不会同时在两边，也不会都不在。**
2. **`used` 中的块 `ref_count ≥ 1`，`free` 中的块 `ref_count == 0`。**
3. **如果 `hash_to_block_id[h] == b`，那么 `blocks[b].hash == h`**，并且块 b 的显存里确实存着这个前缀对应的 KV。反过来不一定成立：可能有多个块的 `hash` 都等于 h，但表里只指向其中一个。
4. **对每条序列，下标 `< num_cached_tokens` 的所有"满块"都已经登记了 hash。** `hash_blocks` 计算下一块的 hash 时，要用到前一块的 hash，靠的就是这一条。

### 3. 块的四种状态

结合 `free/used` 和 `hash` 是否登记，一个块处于下面四种状态之一：

| 状态 | 所在集合 | ref_count | hash | 含义 |
|---|---|---|---|---|
| **FREE_CLEAN** | free | 0 | -1 | 空闲，内容无用 |
| **FREE_CACHED** | free | 0 | 已登记 | 空闲，但显存里的 KV 还有效，可以被"捞回来"复用 |
| **USED** | used | ≥1 | -1 | 正在使用，还没写满或刚写满但尚未登记 |
| **USED_HASHED** | used | ≥1 | 已登记 | 正在使用，已写满并登记，可以被其他序列共享 |

状态转换：

```
                         _allocate_block（popleft，清除旧 hash）
      ┌──────────────┐ ─────────────────────────────────────────►  ┌──────────┐
      │  FREE_CLEAN  │                                              │   USED   │
      └──────────────┘ ◄─────────────────────────────────────────  └──────────┘
             ▲            deallocate（ref 降为 0，未登记 hash）          │
             │                                                          │ hash_blocks
             │ _allocate_block                                          │（块被写满）
             │（被当作新块分配，内容被覆盖）                              ▼
      ┌──────────────┐   allocate 命中（从 free 中 remove，ref=1）  ┌──────────────┐
      │ FREE_CACHED  │ ─────────────────────────────────────────►  │ USED_HASHED  │ ⟲ allocate 命中：ref += 1
      └──────────────┘ ◄─────────────────────────────────────────  └──────────────┘ ⟲ deallocate：ref -= 1（仍 ≥ 1）
                          deallocate（ref 降为 0，保留 hash）
```

关键在于 **FREE_CACHED** 这个状态。块被释放后，hash 登记和显存里的 KV 都还在，所以它仍然可以被命中复用；直到它被 `_allocate_block` 当作新块分配出去，内容才真正失效。这就是 prefix cache 的**惰性淘汰**。

### 4. 链式 hash：`compute_hash`

[block_manager.py:35-41](../../nanovllm/engine/block_manager.py#L35-L41)：

```python
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))   # 前一块的 hash
    h.update(np.array(token_ids).tobytes())      # 本块的 token
    return h.intdigest()
```

```
h0 = H(tokens[0 : bs])
h1 = H(h0 ‖ tokens[bs : 2bs])
h2 = H(h1 ‖ tokens[2bs : 3bs])
```

为什么要把前一块的 hash 链进来？因为 **KV 不只取决于本块的 token，还取决于它前面的全部上下文**。同样的 256 个 token，接在不同前缀后面，算出的 K/V 完全不同。链式 hash 保证 `h_i` 相同意味着从开头到第 i 块的整个前缀都相同，命中的块才可以直接复用。

hash 是 64 位的。`can_allocate` 命中时还会比较本块的 `token_ids` 防止碰撞。前缀部分只靠 hash 链保证，碰撞概率可以忽略。

### 5. 两个底层操作：`_allocate_block` / `_deallocate_block`

[block_manager.py:43-56](../../nanovllm/engine/block_manager.py#L43-L56)：

```python
def _allocate_block(self) -> int:
    block_id = self.free_block_ids.popleft()                 # ① 取队头：最早释放的块
    block = self.blocks[block_id]
    assert block.ref_count == 0
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]                # ② 清除旧的 hash 登记
    block.reset()                                            # ③ ref=1, hash=-1, token_ids=[]
    self.used_block_ids.add(block_id)
    return block_id

def _deallocate_block(self, block_id: int):
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)                     # ④ 放回队尾，hash 保留
```

- **① FIFO 近似 LRU**：从队头取、往队尾放，所以最早释放的块最先被覆盖，最近释放的块在 free 中保留得最久，最可能被命中。
- **② 为什么要判断 `hash_to_block_id.get(block.hash) == block_id`**：因为表里的 h 可能已经指向**别的块**了。典型场景是两条前缀相同的请求在**同一个 prefill step** 被调度：调度时双方的块都还没登记 hash，互相无法命中，于是各自计算、各自分配了块 X 和块 Y。step 结束后，`hash_blocks` 先把 `h → X` 登记进表，随后又被覆盖成 `h → Y`。此时块 X 的 `hash` 仍然是 h，但表里指向的是 Y。如果之后 X 被重新分配时无条件删除 `h`，就会把 Y 的有效登记一起删掉。这个判断保证"只删除属于自己的登记"。
- **③ `reset` 把 `ref_count` 设为 1 而不是 0**：`reset` 只在分配时调用，分配出去就意味着有一个使用者。
- **④ 释放时不清除 hash**：块进入 FREE_CACHED 状态（如果它登记过），这正是惰性淘汰的关键。

### 6. Prefill 分配：`can_allocate` + `allocate`

调用位置在 [scheduler.py:35-45](../../nanovllm/engine/scheduler.py#L35-L45)：只有当 `seq.block_table` 为空时才调用。这有两种情况：序列第一次被调度，或者被抢占后重新调度。chunked prefill 的后续块不会再调用，因为第一次就已经分配好了。

#### `can_allocate`：查命中，算需要几个新块

[block_manager.py:58-73](../../nanovllm/engine/block_manager.py#L58-L73)：

```python
def can_allocate(self, seq) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks                        # 先假设全部要新分配
    for i in range(seq.num_blocks - 1):                    # ① 最后一块不参与匹配
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break                                          # ② 遇到第一个未命中就停
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1                            # ③ 只有 USED_HASHED 的命中才省一个空闲块
    if len(self.free_block_ids) < num_new_blocks:
        return -1                                          # 空闲块不够，不接纳
    return num_cached_blocks                               # 注意 0 也是合法返回值
```

- **① 最后一块永远不复用**：即使整个 prompt 都命中，也要至少重算最后一块。原因是模型必须至少计算 1 个 token，才能得到用于采样下一个 token 的 logits。这里按"块"处理，所以最坏情况下会多算一整块（256 个 token）。
- **② 前缀匹配必须连续**：链式 hash 决定了第 i 块没命中，后面的块不可能命中。
- **③ 命中块对空闲块数量的影响**：
  - 命中的块是 **USED_HASHED**（正在被别人用）：直接共享，不占用空闲块，`num_new_blocks -= 1`；
  - 命中的块是 **FREE_CACHED**（在空闲队列里）：要把它从空闲队列中取出来，**仍然消耗一个空闲名额**，所以不减。
- 返回值约定：`-1` 表示放不下，`≥0` 表示命中的块数。调度器用 `== -1` 判断（[scheduler.py:37](../../nanovllm/engine/scheduler.py#L37)）。

#### `allocate`：真正建立 block_table

[block_manager.py:75-92](../../nanovllm/engine/block_manager.py#L75-L92)：

```python
def allocate(self, seq, num_cached_blocks):
    assert not seq.block_table
    h = -1
    for i in range(num_cached_blocks):                     # ① 先处理命中的块
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1                           #    USED_HASHED：共享
        else:
            block.ref_count = 1                            #    FREE_CACHED：捞回来
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        seq.block_table.append(block_id)
    for i in range(num_cached_blocks, seq.num_blocks):     # ② 再为剩余部分分配新块
        seq.block_table.append(self._allocate_block())
    seq.num_cached_tokens = num_cached_blocks * self.block_size   # ③
```

- **① 和 ② 的顺序很重要**：必须先把命中的 FREE_CACHED 块从空闲队列里取出，再用 `popleft` 分配新块。否则，如果某个命中的块恰好位于队头，它会被 `_allocate_block` 当作新块分配掉，hash 被清除、内容被覆盖，命中就失效了。
- `hash` 会重新计算一遍（`can_allocate` 已经算过）。这是用少量 CPU 开销换取接口简单：`can_allocate` 只返回一个整数，两次调用之间状态不会变化，所以结果一致。
- `free_block_ids.remove(block_id)` 是 deque 上的 O(n) 操作，不过只对命中的 FREE_CACHED 块执行。
- **② 一次性为整个 prompt 分配块**：即使 chunked prefill 会把计算拆到多个 step，块也在第一次就全部分配好。
- **③ `num_cached_tokens` 的设置**：之后 ModelRunner 只计算 `[num_cached_tokens, ...)` 区间的 token，命中部分的 KV 通过 `block_table` 直接从 cache 中读取。

### 7. Decode 追加：`can_append` + `may_append`

[block_manager.py:103-108](../../nanovllm/engine/block_manager.py#L103-L108)，调用位置在 [scheduler.py:60-69](../../nanovllm/engine/scheduler.py#L60-L69)：

```python
def can_append(self, seq) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
def may_append(self, seq):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```

**`% block_size == 1` 的含义**要结合时序来看：

```
step k   的 postprocess：采样得到新 token，append_token → len(seq) = n
step k+1 的 schedule  ：本步要把这个新 token（下标 n-1）的 K/V 写入 cache
                         下标 n-1 是新块的第一个位置 ⇔ (n-1) % bs == 0 ⇔ n % bs == 1
```

也就是说，**新 token 已经追加到序列里了，但它的 K/V 要到本步才写入**。当它正好落在一个新块的第一个位置时，才需要多分配一块。

`can_append` 的写法利用了 Python 中 `bool` 可以当整数用的特点：需要新块时右边是 `True`，即 1，要求至少有 1 个空闲块；不需要时右边是 `False`，即 0，条件总是成立。

空闲块不够时，调度器会抢占其他序列，腾出块后再重试（见 [ContinuousBatching.md](ContinuousBatching.md)）。

### 8. 登记 hash：`hash_blocks`

[block_manager.py:110-120](../../nanovllm/engine/block_manager.py#L110-L120)，在 [scheduler.py:83](../../nanovllm/engine/scheduler.py#L83) 被调用。注意它在 `num_cached_tokens` 更新**之前**调用：

```python
def hash_blocks(self, seq):
    start = seq.num_cached_tokens // self.block_size                               # ① 本步之前已满的块数
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size    #    本步之后已满的块数
    if start == end: return                                                        #    本步没有块被写满
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1          # ② 接上前一块的 hash
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block.update(h, token_ids)                                                 # ③ USED → USED_HASHED
        self.hash_to_block_id[h] = block.block_id                                  # ④ 可能覆盖旧登记
```

- **① 只处理"本步刚写满"的块**：`[start, end)` 就是本步计算之后从未满变成满的那些块。prefill 一次可能写满很多块，decode 每步最多写满一块。未写满的块不登记，因为内容还会继续变化。
- **② 依赖不变式 4**：前一块一定已经登记了 hash，可能是之前某一步登记的，也可能是通过命中得到的块。所以只需从前一块的 hash 接着往下算，不用从头重算。
- **③ 同时保存 `token_ids`**：供日后 `can_allocate` 校验碰撞用。
- **④ 直接覆盖**：如果表里已经有相同的 h（同一个 step 中重复计算的情况），新的登记会覆盖旧的，这正是第 5 节 ② 中那个判断要防范的情况。

`hash_blocks` 对**生成出来的 token** 同样有效。所以多轮对话中，上一轮的回复也能作为下一轮请求的前缀被命中。

### 9. 释放：`deallocate`

[block_manager.py:94-101](../../nanovllm/engine/block_manager.py#L94-L101)，在序列结束（[scheduler.py:91](../../nanovllm/engine/scheduler.py#L91)）和被抢占（[scheduler.py:78](../../nanovllm/engine/scheduler.py#L78)）时调用：

```python
def deallocate(self, seq):
    for block_id in reversed(seq.block_table):     # ① 逆序
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:                   # ② 还有其他序列在用就不释放
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

- **① 逆序释放，让前缀块保留得更久**：

  ```
  block_table = [P0, P1, P2, T]        P = 前缀块，T = 序列特有的尾部块
  逆序释放 → 依次追加到 free 队尾：T, P2, P1, P0
  之后 popleft 时 T 先被覆盖，P0 最后被覆盖
  ```

  越靠前的块，越可能是多个请求共享的内容（比如 system prompt），被再次命中的概率也越高。
- **② 引用计数**：共享的前缀块只有在最后一个使用者释放时，才会回到空闲队列。
- 被抢占的序列也走这条路径。它的块进入 FREE_CACHED 状态，之后重新调度时，`can_allocate` 很可能命中自己原来的块，从而少算一部分。

### 10. 与 Scheduler 的调用关系

| 方法 | 调用位置 | 时机 |
|---|---|---|
| `can_allocate` / `allocate` | `schedule()` prefill 分支 | 序列第一次被调度，或被抢占后重新调度 |
| `can_append` / `may_append` | `schedule()` decode 分支 | 每个 decode step，每条序列一次 |
| `hash_blocks` | `postprocess()` | 每个 step，每条被调度的序列一次，在更新 `num_cached_tokens` 之前 |
| `deallocate` | `preempt()`、`postprocess()` | 被抢占时；结束时 |

---

## 五、完整状态追踪

block_size = 4，共 6 个物理块。用 `h0`、`h1` 等表示链式 hash。

**初始**：`free = [0,1,2,3,4,5]`，`used = {}`，`map = {}`

**Step 1：A prefill**。A 的 prompt 有 10 个 token：`[S0..S7, a0, a1]`

```
can_allocate(A)：块 0 的 h0 不在 map 中 → 立即停止，命中 0 块，需要 3 个新块，free 有 6 个 → 返回 0
allocate(A)    ：_allocate_block × 3 → A.block_table = [0,1,2]，free = [3,4,5]
计算 10 个 token，采样得到 a2
hash_blocks(A) ：start = 0//4 = 0，end = 10//4 = 2 → 登记块 0（h0）和块 1（h1）
                 map = {h0:0, h1:1}；块 2 只有 2 个 token，不登记
postprocess    ：A.num_cached_tokens = 10，append a2 → len(A) = 11
```

**Step 2：B prefill**（B 共享 8 个 token 的前缀；假设它因 token 预算在这一步才被调度）。B 的 prompt：`[S0..S7, b0, b1, b2]`，共 11 个 token

```
can_allocate(B)：i=0 时 h0 命中块 0（USED）→ 新块需求 3→2
                 i=1 时 h1 命中块 1（USED）→ 新块需求 2→1；只检查到 num_blocks-1 = 2，循环结束
                 free 有 3 个 ≥ 1 → 返回 2
allocate(B)    ：块 0、块 1 的 ref 变为 2；为第 2 块分配块 3
                 B.block_table = [0,1,3]，free = [4,5]，B.num_cached_tokens = 8
计算只针对 b0..b2 这 3 个 token；attention 通过 block_table 从块 0、1 读取前缀的 K/V
hash_blocks(B) ：start = 8//4 = 2，end = 11//4 = 2 → 没有块被写满
postprocess    ：append b3 → len(B) = 12
（这一步 A 没有被调度：prefill 优先，A 的 decode 暂停一步）
```

**Step 3：decode A、B**

```
A：len = 11，11 % 4 = 3 → 不需要新块；写入 a2（下标 10，块 2 的第 2 个位置）
B：len = 12，12 % 4 = 0 → 不需要新块；写入 b3（下标 11，块 3 的第 3 个位置）→ 块 3 写满
hash_blocks(B) ：start = 2，end = 12//4 = 3 → 登记块 3：h2B = H(h1 ‖ [b0,b1,b2,b3])
map = {h0:0, h1:1, h2B:3}
postprocess    ：A append a3 → len 12；B append b4 → len 13
```

**Step 4：decode A、B；A 结束**

```
A：len = 12 → 不需要新块；写入 a3（下标 11）→ 块 2 写满
B：len = 13，13 % 4 == 1 → may_append：popleft 得到块 4 → B.block_table = [0,1,3,4]，free = [5]
hash_blocks(A) ：登记块 2：h2A = H(h1 ‖ [a0,a1,a2,a3])
                 map = {h0:0, h1:1, h2B:3, h2A:2}
A 采样到 EOS → deallocate(A)，按逆序处理 [2,1,0]：
    块 2：ref 1→0 → 放回 free，进入 FREE_CACHED（hash h2A 仍在 map 中）
    块 1：ref 2→1（B 还在用）
    块 0：ref 2→1
free = [5, 2]
```

**Step 5：C prefill**（多轮对话：C 的 prompt 是 A 的完整对话再加一个新 token）。C 的 prompt：`[S0..S7, a0, a1, a2, a3, c0]`，共 13 个 token，需要 4 块

```
can_allocate(C)：h0 命中块 0（USED）→ 需求 4→3
                 h1 命中块 1（USED）→ 需求 3→2
                 h2A 命中块 2（FREE_CACHED）→ 需求不变，仍为 2
                 检查到 num_blocks-1 = 3 为止 → 命中 3 块；free 有 2 个 ≥ 2 → 返回 3
allocate(C)    ：块 0、块 1 的 ref 变为 2
                 块 2：从 free 中 remove，ref = 1 → free = [5]（捞回来，注意要先于下面的 popleft）
                 第 3 块：popleft 得到块 5 → free = []
                 C.block_table = [0,1,2,5]，C.num_cached_tokens = 12 → 只计算 c0 这 1 个 token
```

A 生成的 `a2`、`a3` 也被复用了。这就是 `hash_blocks` 对生成 token 同样有效的意义。

**Step 6~8：decode B、C**，各自写满块 4、块 5 并登记 hash（`h3B → 4`，`h3C → 5`）。两者的 len 都变为 17。

**Step 9：显存不足，抢占**

```
running = [B, C]，free = []
B：len = 17，17 % 4 == 1 → 需要新块，can_append 失败 → preempt(running.pop()) 抢占 C
   deallocate(C)，按逆序处理 [5,2,1,0]：
       块 5：ref 1→0 → free = [5]          （FREE_CACHED，h3C）
       块 2：ref 1→0 → free = [5, 2]       （FREE_CACHED，h2A）
       块 1：ref 2→1；块 0：ref 2→1
   C 回到 waiting 队头
   can_append(B) 成功 → may_append：popleft 得到块 5
       _allocate_block 发现 map[h3C] == 5 → 删除 h3C 的登记，块 5 被覆盖
   free = [2]
```

由于逆序释放，C 的尾部块 5 先被覆盖，前缀块 2 保留了下来。

C 不能马上重新调度：此时它需要 5 块，命中的块 0、1 正被 B 使用，不占名额；块 2 是 FREE_CACHED，仍占 1 个名额。所以还需要 3 个空闲块，而 free 里只有 1 个，`can_allocate` 返回 -1。等 B 结束、块被释放后：

```
free = [2, 5, 4, 3, 1, 0]     （B 的块按逆序 [5,4,3,1,0] 追加到队尾）
can_allocate(C)：h0、h1、h2A 都命中（块 0、1、2 都是 FREE_CACHED），h3C 已被删除 → 命中 3 块
                 → 只需要重新计算下标 12..16 这 5 个 token，而不是全部 17 个
```

> 这一节的每一步都用真实的 `BlockManager` 代码模拟验证过（block_size = 4，6 个块）。

---

## 六、地址翻译层：slot_mapping 和 block_tables

逻辑层只有整数。ModelRunner 每步把它们翻译成两个 GPU 张量，一个用于写，一个用于读。

**写：`slot_mapping`，每个新 token 对应一个槽位号**

```
slot = block_table[pos // block_size] * block_size + pos % block_size
```

把物理 cache 看成一个长度为 `num_blocks × block_size` 的一维数组，slot 就是这个一维数组里的下标。

- Prefill：[model_runner.py](../../nanovllm/engine/model_runner.py#L152-L162) 按块计算区间 `[num_cached_tokens, num_cached_tokens + num_scheduled_tokens)` 内每个 token 的槽位；
- Decode：[model_runner.py](../../nanovllm/engine/model_runner.py#L181) 计算 `block_table[-1] * block_size + last_block_num_tokens - 1`。

**读：`block_tables`，每条序列一行**

[`prepare_block_tables`](../../nanovllm/engine/model_runner.py#L123-L127) 把各序列的 block_table 用 -1 补齐成 `[num_seqs, max_blocks]` 的矩形张量，交给 attention kernel。

---

## 七、物理层的读写

### 1. 写入：Triton kernel `store_kvcache`

[attention.py](../../nanovllm/layers/attention.py#L10-L40)：

```python
@triton.jit
def store_kvcache_kernel(...):
    idx = tl.program_id(0)                       # 每个 token 一个 program
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return                        # CUDA Graph 的 padding 行，跳过
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))   # D = kv_heads × head_dim
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)       # 写到第 slot 个槽位
```

`k_cache` 的形状是 `[num_blocks, block_size, kv_heads, head_dim]`，内存连续，所以槽位 `slot` 的起始地址就是 `slot × D`。kernel 里断言了 `k_cache.stride(1) == D`，保证这个线性寻址成立。

每一层的 `Attention.forward` 第一件事就是写 cache，然后再计算 attention。

### 2. 读取：FlashAttention 的 `block_table` 参数

[attention.py](../../nanovllm/layers/attention.py#L64-L75)：

```python
if context.is_prefill:
    if context.block_tables is not None:          # 有前缀（prefix cache 命中或 chunked prefill 的后续块）
        k, v = k_cache, v_cache                   # 改从分页 cache 读取整条序列的 K/V
    o = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...,
                               block_table=context.block_tables)
else:                                             # decode
    o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                cache_seqlens=context.context_lens,
                                block_table=context.block_tables)
```

| 场景 | K/V 从哪里读 | 为什么 |
|---|---|---|
| prefill，无前缀 | 本步刚算出的 `k, v`（连续张量） | 所需 K/V 全部在本步，直接用最快 |
| prefill，有前缀 | `k_cache / v_cache` + `block_table` | 前缀的 K/V 只存在于 cache 中，要通过 block table 找到 |
| decode | `k_cache / v_cache` + `block_table` + `context_lens` | 历史 K/V 全部在 cache 中 |

kernel 内部会按 `block_table[seq][pos // block_size]` 找到物理块，在块内按 `pos % block_size` 取数据。这一步就是 PagedAttention 中"按页表跳着读"的部分。

---

## 八、与 vLLM 原版的差异

| 方面 | vLLM | nano-vllm |
|---|---|---|
| 块大小 | 默认 16 | 256（FlashAttention 分页接口的要求） |
| attention kernel | 自研 paged attention kernel，或 FlashAttention / FlashInfer 等后端 | 直接用 `flash_attn` 的 `block_table` 参数 |
| 写 cache | 自研 CUDA kernel `reshape_and_cache` | 约 20 行的 Triton kernel |
| 共享方式 | prefix sharing + copy-on-write（用于并行采样、beam search） | 只有 prefix sharing，没有 CoW |
| 显存不足 | recompute，或 swap 到 CPU | 只有 recompute |
| 淘汰策略 | LRU 等 | FIFO 空闲队列 + 逆序释放，近似 LRU |

**块大小为 256 的影响**：

- 每条序列平均浪费半块，也就是约 128 个 token 的 KV（Qwen3-0.6B 上约 14 MB）。短请求很多时，这部分浪费比 vLLM 明显。
- prefix cache 的粒度也更粗：前缀至少要有完整的 256 个 token 才可能命中；"最后一块不复用"的规则最坏会多算 256 个 token。
- 同一个 prefill step 中前缀相同的多条请求无法互相共享（第四节第 5 小节 ② 的场景），只有之后到来的请求才能命中。
- 好处是块数少，block table 短，CPU 端的管理开销也低。
