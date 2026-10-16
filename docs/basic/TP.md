# 张量并行（Tensor Parallelism）

本文说明 nano-vllm 如何把一个模型切到多张 GPU 上运行。本文只讲整体设计，每个函数的细节（切分的下标怎样计算、bias 为什么只加一次、共享内存的消息格式等）直接写在源码注释中，文中会给出对应的位置。

模型维度以 Qwen3-8B 为例：hidden = 4096，32 个 q 头，8 个 kv 头，head_dim = 128，MLP 中间维度 12288，36 层，词表 151936。下文用 N 表示 TP 的卡数。

---

## 一、TP 是什么

张量并行（Tensor Parallelism）把**每一层的权重矩阵切开**，分到多张 GPU 上。每张卡只存、只算其中一部分，必要时通过通信把结果拼起来。所有卡**同时**处理同一批 token 的同一层。

它和另外两种并行的区别：

| 方式 | 切什么 | 每张卡做什么 | 通信 |
|---|---|---|---|
| 数据并行（DP） | 不切模型，切请求 | 各自用完整模型处理不同的请求 | 推理时基本没有 |
| 流水线并行（PP） | 按层切：卡 0 放前 18 层，卡 1 放后 18 层 | 依次接力 | 层与层之间传激活 |
| **张量并行（TP）** | 切每一层内部的矩阵 | 同时算同一层的不同部分 | **每层都要通信** |

## 二、为什么推理需要 TP

1. **放得下**：Qwen3-8B 的 bf16 权重约 16GB，更大的模型（70B 约 140GB）单卡根本放不下。TP = N 时，每张卡只存 1/N 的权重；
2. **decode 更快**：decode 每步都要把全部权重从显存读一遍，瓶颈在访存带宽（见 [ContinuousBatching.md](ContinuousBatching.md)）。TP = N 时每张卡只读 1/N 的权重，N 张卡的带宽同时工作，单步延迟理论上可以接近 1/N。DP 做不到这一点，因为每个请求仍然只用到一张卡；
3. **KV cache 更大**：每张卡只存自己负责的那部分注意力头的 KV（见第六节），同样的显存可以容纳更多 token，batch 也就能开得更大。

**代价**：每层要做两次 all-reduce。通信量本身不大（每次是 `token 数 × hidden`），但次数多，对延迟很敏感。所以 TP 一般只在单机内、通过 NVLink 互联的卡之间使用。

---

## 三、列并行与行并行

线性层计算 `Y = X · Wᵀ`。PyTorch 中 `W` 的形状是 `[out, in]`。切 `W` 有两种方式。

### 1. 列并行（Column Parallel）：沿输出维度切

```
W [out, in] 按 dim 0 切成 N 份：W_0, W_1, …        每张卡 [out/N, in]

卡 i：Y_i = X · W_iᵀ         输入 X：完整的（每张卡都有）
                             输出 Y_i：[tokens, out/N]，只有输出特征的第 i 段
```

每张卡得到**输出的一部分列**，计算本身不需要通信。如果后面需要完整的 Y，就要做 **all-gather**（把各卡的段拼起来）。

### 2. 行并行（Row Parallel）：沿输入维度切

```
W [out, in] 按 dim 1 切成 N 份：                    每张卡 [out, in/N]

卡 i：Y_i = X_i · W_iᵀ       输入 X_i：[tokens, in/N]，只有输入特征的第 i 段
                             输出 Y_i：[tokens, out]，形状完整，但只是"部分和"

Y = Y_0 + Y_1 + … + Y_{N−1}  需要 all-reduce（求和）
```

每张卡得到**形状完整但只是部分和**的输出，需要 **all-reduce** 把各卡的结果加起来。

### 3. 关键组合：列并行后面接行并行，中间不需要通信

列并行的**输出**正好是按特征切开的 `[tokens, out/N]`，这恰好是行并行需要的**输入** `X_i`。所以：

```
完整 X ──► 列并行 ──► Y_i（切开的） ──► 逐元素操作 ──► 行并行 ──► 部分和 ──► all-reduce ──► 完整输出
(每卡都有)     无通信                     无通信             无通信            唯一一次通信
```

两个矩阵乘只需要**一次 all-reduce**。这就是 Megatron-LM 的经典切法，Transformer 中的两个子模块都可以这样组织：

- **MLP**：gate_up（列）→ SiLU × up（逐元素）→ down（行）→ all-reduce
- **Attention**：qkv（列，按头切）→ 每张卡独立算自己那些头的注意力 → o_proj（行）→ all-reduce

中间那一步之所以能在切开的数据上直接做，是因为它对特征维度是**逐元素**或**按头独立**的：SiLU、乘法逐元素进行，注意力的各个头之间互不相干。

---

## 四、常用的集合通信操作

| 操作 | 作用 | nano-vllm 中的使用 |
|---|---|---|
| **all-reduce** | 每张卡给出一个同形状的张量，求和后**每张卡都**拿到总和 | 行并行层（o_proj、down_proj）；词表并行的 embedding |
| **gather** | 每张卡给出一段，**只有目标卡**拿到拼接结果 | LMHead 把 logits 汇集到 rank 0 |
| all-gather | 每张卡给出一段，**每张卡都**拿到拼接结果 | 未使用 |
| reduce-scatter | 求和之后切开，每张卡只拿一段 | 未使用（序列并行会用到：all-reduce = reduce-scatter + all-gather） |
| broadcast | 一张卡把数据发给所有卡 | 未使用。控制信息用共享内存传递（见第七节），张量的"广播"由 all-reduce 顺带完成（见下一小节） |
| barrier | 所有卡在此汇合，不传数据 | 建立共享内存、退出时同步 |

### 输入 X 是怎样到达每个 rank 的

列并行要求每个 rank 都有**完整的** X。nano-vllm 中没有任何一次 broadcast，X 是通过下面两种方式出现在每个 rank 上的：

| X 出现的位置 | 怎样到达每个 rank |
|---|---|
| **模型的输入**（token id） | rank 0 通过**控制面**（第七节）把本步的 `seqs` 发给所有 rank，每个 rank 自己构造出相同的 `input_ids`；embedding 查表后做一次 **all-reduce**，每个 rank 都得到完整的 `[tokens, hidden]` |
| **每个列并行层的输入**（hidden states） | 前一个行并行层的 **all-reduce** 本身就让每个 rank 拿到了完整结果，它直接就是下一层的输入；中间的 norm、残差在每个 rank 上各算一遍 |

也就是说，all-reduce 同时完成了两件事：把部分和加起来，并把完整结果"广播"给所有 rank。所以整个前向中不需要额外的 broadcast。

---

## 五、并行线性层

[linear.py](../../nanovllm/layers/linear.py) 把第三节的两种切法实现为一组线性层。每一层只按本 rank 那一片的大小分配权重，加载时由各自的 `weight_loader` 从完整的 checkpoint 中切出这一片（见本节第 3 小节）。

### 1. 各类并行线性层

| 类 | 对应第三节 | 每个 rank 的权重 | 前向的通信 | Qwen3 中用于 |
|---|---|---|---|---|
| `ColumnParallelLinear` | 列并行 | `[out/N, in]` | 无 | 下面两个类的基类 |
| `MergedColumnParallelLinear` | 列并行，权重由多个矩阵拼成 | `[Σout/N, in]` | 无 | gate_up_proj |
| `QKVParallelLinear` | 列并行，权重由 q/k/v 拼成，按头切 | `[(H + 2·Hkv)/N · D, in]` | 无 | qkv_proj |
| `RowParallelLinear` | 行并行 | `[out, in/N]` | **all-reduce** | o_proj、down_proj |
| `ReplicatedLinear` | 不切 | `[out, in]` | 无 | 未使用 |

另外，**行并行的 bias 只由 rank 0 加**，否则 all-reduce 后会被加 N 次（见 `RowParallelLinear.forward` 的注释）。

### 2. 合并权重的列并行：`MergedColumnParallelLinear` 和 `QKVParallelLinear`

#### (1) 为什么要合并

HuggingFace 的 checkpoint 中，`q_proj`、`k_proj`、`v_proj` 是三个独立的矩阵，`gate_proj`、`up_proj` 是两个。nano-vllm 把它们拼成一个矩阵，三次（或两次）GEMM 合成一次，减少 kernel 数量。

[qwen3.py](../../nanovllm/models/qwen3.py) 中的 `packed_modules_mapping` 记录了这种对应关系：

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

[loader.py](../../nanovllm/utils/loader.py) 读到 `...gate_proj.weight` 时，会找到 `gate_up_proj` 的参数，调用它的 `weight_loader(param, 权重, 0)`。第三个参数 `shard_id` 指明这是第几个子矩阵（加载的完整流程见第 3 小节）。

#### (2) 为什么不能用普通的列并行来切

以 gate_up、TP = 2 为例，拼接后的完整矩阵是 `[gate 12288 行 | up 12288 行]`。如果像 `ColumnParallelLinear` 那样把整个矩阵按 dim 0 切成两半：

```
❌ 整体切：    rank 0 = [gate 全部 12288 行]          rank 1 = [up 全部 12288 行]
   → rank 0 只有 gate，没有 up，无法计算 SiLU(gate) × up
```

正确的做法是**每个子矩阵分别切成 N 份，再把每个子矩阵的第 i 份拼在一起放到 rank i**：

```
✅ 分别切再拼：rank 0 = [gate 前 6144 行 | up 前 6144 行]
              rank 1 = [gate 后 6144 行 | up 后 6144 行]
   → 每个 rank 上的 gate 和 up 对应同一段中间维度，可以直接逐元素相乘
```

`QKVParallelLinear` 同理，只是 q、k、v 三段大小不同，并且按头的边界切：每个 rank 拿到的是**完整的若干个头**，attention 才能按头独立计算。Qwen3-8B、TP = 2 时，每个 rank 的 qkv 权重是 `[q：16 头 × 128 = 2048 行 | k：4 头 × 128 = 512 行 | v：4 头 × 128 = 512 行]`。这也是头数和 kv 头数必须能被 N 整除的原因。

### 3. 加载：每一层怎样从完整的 checkpoint 中切出自己那一片

每一层在构造时，把自己的 `weight_loader` 方法挂到权重参数上（`LinearBase.__init__` 中的 `self.weight.weight_loader = self.weight_loader`）。[loader.py](../../nanovllm/utils/loader.py) 因此只需要一个通用的循环，具体切哪部分完全交给参数自己：

```python
for weight_name in checkpoint:                          # 例如 "...mlp.gate_proj.weight"
    if 名字命中 packed_modules_mapping:                  # q_proj / k_proj / v_proj / gate_proj / up_proj
        param = 改写后的参数（如 gate_up_proj.weight）
        param.weight_loader(param, loaded_weight, shard_id)
    else:
        param = model.get_parameter(weight_name)
        getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)   # 没挂的（如 RMSNorm）完整拷贝
```

两个参数的含义：

- **`param`**：目标。模型中已分配好的参数，形状是**本 rank 那一片**；
- **`loaded_weight`**：来源。从 checkpoint 读出的权重，形状是**完整的**。

每个 `weight_loader` 做的都是同一件事：**从 `loaded_weight` 中取出本 rank 需要的部分，原地拷贝进 `param` 的对应位置**。各层的区别只在于"取哪部分、放到哪里"（rank i）：

| 层 | 从 `loaded_weight` 中取 | 放进 `param` 的哪里 |
|---|---|---|
| `ReplicatedLinear`、RMSNorm（`default_weight_loader`） | 全部 | 全部 |
| `ColumnParallelLinear` | 沿 dim 0（输出）的第 i 段 | 全部 |
| `RowParallelLinear` | weight：沿 dim 1（输入）的第 i 段；bias：全部 | 全部 |
| `MergedColumnParallelLinear` | 子矩阵（gate 或 up）沿 dim 0 的第 i 段 | 按 `shard_id` 定位：gate → 前半，up → 后半 |
| `QKVParallelLinear` | 子矩阵（q、k 或 v）按头的第 i 段 | 按 `shard_id` 定位：q → `[0, q)`，k → `[q, q+kv)`，v → `[q+kv, q+2kv)` |
| `VocabParallelEmbedding` / `ParallelLMHead` | 沿词表（dim 0）的第 i 段 | 全部 |

可以归纳为两类：

- **普通的层**：只需要回答"取哪部分"，然后整体放进 `param`；
- **合并的层**（Merged、QKV）：一个 `param` 对应 checkpoint 中的多个权重，要被调用多次。每次除了"取哪部分"，还要用 `shard_id` 回答"放到 `param` 的哪个位置"。

例：TP = 2 时，rank 1 的 `gate_up_proj.weight`（`[12288, 4096]`）被调用两次才填满：

```
① 读到 gate_proj.weight [12288, 4096]，shard_id = 0 → 取 gate 的第 2 段 [6144, 4096] → 写进 param[0:6144]
② 读到 up_proj.weight   [12288, 4096]，shard_id = 1 → 取 up 的第 2 段   [6144, 4096] → 写进 param[6144:12288]
```

两次之后，本 rank 的 `param` 恰好是 `[gate 第 2 段 | up 第 2 段]`，正是 `SiluAndMul` 需要的排布。

---

## 六、线性层之外的部分

### 1. 为什么要分开设计

第五节的线性层都是"权重矩阵乘输入"，只要决定沿输入还是输出切即可。模型中其余的部分不是这种形式，切法要根据各自的计算特点来定：

| 部分 | 不同之处 | 因此的切法 |
|---|---|---|
| embedding | 不是矩阵乘，而是**按 token id 查表** | 按词表切，再 all-reduce |
| attention | **没有权重**，但有按头的结构，还要读写 **KV cache** | 跟随 qkv_proj 按头切，KV cache 也按头切 |
| norm | 要对**整个 hidden 维度**求均方根，切开就算不对 | 不切，每个 rank 各算一遍 |
| rotary、SiLU × up | 逐头或逐元素 | 不切，直接在切开的数据上算 |
| lm_head | 输出是词表，且**只有 rank 0 采样** | 按词表切，只 gather 到 rank 0 |

### 2. 逐层汇总

| 层 | 切法 | 每个 rank 的参数 | 通信 | 代码 |
|---|---|---|---|---|
| embed_tokens | 按词表切 | `[vocab/N, hidden]` | **all-reduce** | [embed_head.py](../../nanovllm/layers/embed_head.py) `VocabParallelEmbedding` |
| RMSNorm | 复制 | 完整 | 无 | [qwen3.py](../../nanovllm/models/qwen3.py) `Qwen3DecoderLayer` |
| qkv_proj | 列并行，按头切 | 本 rank 的 q/k/v 头 | 无 | [linear.py](../../nanovllm/layers/linear.py) `QKVParallelLinear` |
| q_norm / k_norm、rotary | 复制，逐头计算 | 完整 | 无 | |
| attention | 按头切 | 无参数 | 无 | [qwen3.py](../../nanovllm/models/qwen3.py) `Qwen3Attention` |
| KV cache | 按 kv 头切 | 本 rank 的 kv 头 | 无 | [model_runner.py](../../nanovllm/engine/model_runner.py) `allocate_kv_cache` |
| o_proj | 行并行 | `[hidden, H·D/N]` | **all-reduce** | `RowParallelLinear` |
| gate_up_proj | 列并行 | `[2·inter/N, hidden]` | 无 | `MergedColumnParallelLinear` |
| SiluAndMul | 复制，逐元素 | 无参数 | 无 | |
| down_proj | 行并行 | `[hidden, inter/N]` | **all-reduce** | `RowParallelLinear` |
| lm_head | 按词表切 | `[vocab/N, hidden]` | **gather 到 rank 0** | [embed_head.py](../../nanovllm/layers/embed_head.py) `ParallelLMHead` |

一个 decoder 层中的数据流（TP=2）：

```
hidden [T, 4096]             ← 每个 rank 都有完整的一份
  │ input_layernorm           复制
  │ qkv_proj                  列并行 → 本 rank 16 个 q 头、4 个 kv 头
  │ q/k norm、rotary          复制，逐头
  │ attention + KV cache      本 rank 的头独立计算，KV 只写本 rank 的 kv 头
  │ o_proj                    行并行 → all-reduce ①
hidden [T, 4096]             ← 又是每个 rank 都有完整的一份
  │ post_attention_layernorm  复制
  │ gate_up_proj              列并行 → [T, 12288]（gate 6144 | up 6144）
  │ SiluAndMul                逐元素 → [T, 6144]
  │ down_proj                 行并行 → all-reduce ②
hidden [T, 4096]
```

一次前向的通信共 **1（embedding）+ 2 × 层数 + 1（lm_head）** 次，Qwen3-8B 是 74 次。

**并行度的限制**：q 头数、kv 头数、词表大小、MLP 中间维度都必须能被 N 整除。Qwen3-8B 只有 8 个 kv 头，所以 N 最大为 8，这也是 [config.py](../../nanovllm/config.py) 中 `tensor_parallel_size <= 8` 的现实背景。

**KV cache 的切分**：每个 rank 只存自己那几个 kv 头，每个块的字节数变为 1/N，同样的显存能分出 N 倍的块数。所有 rank 共用 rank 0 的 Scheduler 分配的同一张 block_table，只是每个块中存的是各自那几个头的数据。

---

## 七、nano-vllm 的整体设计：控制面与数据面

切好模型只是第一步，还要让 N 个进程步调一致地执行。nano-vllm 采用 **SPMD** 模式：每个 rank 都有一个完整的 `ModelRunner`，**执行同一个方法、同一条代码路径**，只是持有的权重分片不同。为此需要两套机制：

| | 控制面 | 数据面 |
|---|---|---|
| 解决的问题 | 这一步**做什么**：调用哪个方法、本批有哪些序列、每条算几个 token、KV 写到哪些块 | 计算过程中**交换张量** |
| 传输的内容 | CPU 上的 Python 对象（方法名、`seqs`） | GPU 上的张量 |
| 方向 | 只有 rank 0 → 其他 rank | 所有 rank 之间 |
| 机制 | 一块 1MB 共享内存 + 每个子进程一个 Event | NCCL（`torch.distributed`） |
| 代码 | `call` / `write_shm` / `read_shm` / `loop` | 并行层 forward 中的 `all_reduce`、`gather` |

两套机制在一次 `run` 中的位置如下：① 是控制面，② 是数据面。

```
          rank 0（主进程）                                         rank 1..N−1（子进程）
 ┌──────────────────────────────────┐                     ┌──────────────────────────────────┐
 │ LLMEngine                        │                     │                                  │
 │   Scheduler（只有一份）           │                     │                                  │
 │   tokenizer                      │                     │                                  │
 │       │                          │                     │                                  │
 │       ▼                          │                     │ loop():                          │
 │ ModelRunner.call("run",          │  ① 控制面            │   event.wait()                   │
 │          seqs, is_prefill)       │ ─────────────────► │   read_shm()：反序列化            │
 │   write_shm()：pickle → 共享内存   │  共享内存 + Event    │   call("run", …)：只在本地执行     │
 │   → set 每个子进程的 Event         │                     │       │                          │
 │       │                          │                     │       ▼                          │
 │       ▼                          │                     │                                  │
 │ run(seqs, is_prefill)            │                     │ run(seqs, is_prefill)            │
 │   prepare_* 构造输入              │                     │   prepare_* 构造同样的输入         │
 │   model(...)  ◄══════════════════╪══ ② 数据面：NCCL ═══╪══►  model(...)                   │
 │                                  │  每层 all-reduce     │                                  │
 │                                  │  lm_head gather     │                                  │
 │   Sampler（只在 rank 0）          │                     │   返回 None                       │
 │       │                          │                     │       │                          │
 │       ▼                          │                     │       ▼                          │
 │ Scheduler.postprocess            │                     │ 回到 loop，等待下一条消息          │
 └──────────────────────────────────┘                     └──────────────────────────────────┘
```

### 1. 进程模型

```
主进程
 ├─ LLMEngine
 │   ├─ Scheduler、BlockManager、tokenizer      ← 只有一份
 │   └─ ModelRunner(rank=0)                    ← GPU 0；同时负责调度和计算
 ├─ 子进程：ModelRunner(rank=1)                 ← GPU 1
 └─ 子进程：ModelRunner(rank=N−1)               ← GPU N−1
```

启动顺序（[llm_engine.py](../../nanovllm/engine/llm_engine.py) 的 `__init__`）：

1. 用 `spawn` 启动 rank 1 ~ N−1，子进程的入口就是 `ModelRunner` 的构造函数；
2. 在主进程中构造 rank 0；
3. 所有 rank 在 `init_process_group` 汇合，之后各自加载自己的权重分片、warmup、分配 KV cache、录 CUDA Graph。这些步骤中有 all-reduce，但因为每个 rank 执行的是同一段 `__init__`，通信顺序天然一致；
4. 建立共享内存，子进程进入 `loop()` 等待指令；
5. 最后创建 Scheduler，它需要 rank 0 算出的 KV cache 块数。

### 2. 一步的执行过程

```
时间 │ rank 0（主进程）                               │ rank 1..N−1（子进程）
─────┼─────────────────────────────────────────────────┼──────────────────────────────────
 t0  │ Scheduler.schedule() → seqs                      │ 阻塞在 event.wait()
 t1  │ call("run", seqs, is_prefill)                    │
     │   ① 控制面：pickle → 写共享内存 → set Event       │
 t2  │   run()：prepare_* 构造输入                       │ 醒来 → 读共享内存 → run()：构造同样的输入
 t3  │   前向                                            │ 前向
     │   ② 数据面：每层两次 all-reduce ◄══════════════► │
     │            lm_head 的 gather    ◄══════════════► │
 t4  │   Sampler → token_ids                             │ 返回 None，回到 loop 等待
     │ postprocess：更新序列状态                          │
```

各 rank 收到同一份 `seqs`、用同样的代码构造输入、按同样的顺序调用各层，所以数据面的集合通信**天然一一对应**，不需要额外协调。这也是 CUDA Graph 在 TP 下能工作的原因：各 rank 选中同一个档位的图，图中录下的 all-reduce 能够对上（见 [cuda-graph.md](cuda-graph.md)）。

| 只在 rank 0 上做 | 每个 rank 都做 |
|---|---|
| 调度（Scheduler、BlockManager） | 构造输入（`prepare_*`），CPU 工作重复 N 遍 |
| 分词、反分词 | 模型前向 |
| 采样（只有 rank 0 拿到完整 logits） | |

子进程每步拿到的是**新反序列化出的** Sequence 对象，用完即丢，修改它不会传回 rank 0。所以序列状态只能由 rank 0 维护。`Sequence.__getstate__` 只传子进程构造输入所需的字段，decode 时只传 `last_token`，见 [Sequence.md](Sequence.md)。

### 3. 控制面的一个隐含约束

控制面只有 rank 0 → 子进程的通知，子进程从不回复"已读"。rank 0 之所以不会在子进程读完之前覆盖消息，是因为**数据面间接完成了同步**：子进程总是先读完消息再开始执行方法，而方法中的集合通信要等所有 rank 都到达才会完成。所以 rank 0 的方法返回时，子进程一定已经读完了。

由此得出一个约束：**每个通过 `call` 调用的方法都必须至少包含一次集合通信**。目前的 `run`（前向中有 all-reduce）和 `exit`（有 barrier）都满足。以后新增方法时（例如推测解码的 `run_spec`）需要注意这一点。

### 4. 设计取舍与局限

| 设计 | 好处 | 局限 |
|---|---|---|
| 控制面用共享内存，而不是 NCCL broadcast | 直接传 Python 对象，不用先转成 GPU 张量；实现只有几十行 | 只能单机；容量固定 1MB |
| 只有 rank 0 运行 Scheduler | 调度只做一次，状态只有一份 | 调度和 GPU 计算串行，调度期间所有 GPU 空闲 |
| `call` 统一广播方法名 | LLMEngine 的代码与单卡几乎一样 | 隐含约束：每个方法都要有集合通信 |
| 每个 rank 各自构造输入 | 不用把输入张量从 rank 0 传过去 | CPU 上的准备工作重复 N 遍 |
| 共享内存名 `"nanovllm"`、端口 `2333` 固定 | 简单 | 同一台机器上不能同时运行两个 TP > 1 的实例 |
| Scheduler 使用 rank 0 算出的 KV 块数 | 简单 | 各卡空闲显存不同时，其他 rank 的 KV cache 可能更小，没有检查 |

各函数的具体实现（汇合与建立共享内存的顺序、消息格式、Event 的复位、退出时的释放顺序）见 [model_runner.py](../../nanovllm/engine/model_runner.py) 中 `__init__`、`call`、`loop`、`read_shm`、`write_shm`、`exit` 的注释。

---

## 八、用并行层搭建 Qwen3

前面几节分别讲了并行线性层（第五节）和其余各层的切法（第六节）。本节说明 [qwen3.py](../../nanovllm/models/qwen3.py) 怎样把它们组装成完整的 `Qwen3ForCausalLM`。模型结构与 HuggingFace 的 Qwen3 一一对应，区别只在于：**凡是有大权重矩阵的地方，都换成了对应的并行层**。各模块的实现细节见 qwen3.py 中的注释。

### 1. 模块树

```
Qwen3ForCausalLM                                                每个 rank 上的权重（Qwen3-8B，TP = 2）
  ├── model: Qwen3Model
  │     ├── embed_tokens: VocabParallelEmbedding                [75968, 4096]    按词表切 → all-reduce
  │     ├── layers: ModuleList × 36
  │     │     └── Qwen3DecoderLayer
  │     │           ├── input_layernorm: RMSNorm                [4096]           复制
  │     │           ├── self_attn: Qwen3Attention
  │     │           │     ├── qkv_proj: QKVParallelLinear       [3072, 4096]     列并行，按头切
  │     │           │     ├── q_norm / k_norm: RMSNorm          [128]            复制
  │     │           │     ├── rotary_emb: RotaryEmbedding       cos/sin 表        复制，所有层共享一张
  │     │           │     ├── attn: Attention                   无参数           KV cache 只存本 rank 的 4 个 kv 头
  │     │           │     └── o_proj: RowParallelLinear         [4096, 2048]     行并行 → all-reduce
  │     │           ├── post_attention_layernorm: RMSNorm       [4096]           复制
  │     │           └── mlp: Qwen3MLP
  │     │                 ├── gate_up_proj: MergedColumnParallelLinear   [12288, 4096]   列并行
  │     │                 ├── act_fn: SiluAndMul                无参数
  │     │                 └── down_proj: RowParallelLinear      [4096, 6144]     行并行 → all-reduce
  │     └── norm: RMSNorm                                       [4096]           复制
  └── lm_head: ParallelLMHead                                   [75968, 4096]    按词表切 → gather 到 rank 0
```

整个模型的并行结构就是**两组"列并行 → 行并行"的配对**，首尾再加上词表并行。第五节中的四种线性层，Qwen3 用到了三种（没有用 `ReplicatedLinear`）。一个 decoder 层内部的数据流见第六节第 2 小节。

### 2. 各模块怎样组装

| 模块 | 由什么组成 | 与 TP 相关的要点 |
|---|---|---|
| **Qwen3Attention** | `QKVParallelLinear` → q/k norm → rotary → `Attention` → `RowParallelLinear` | **整个模型中唯一显式感知 TP 的地方**：根据 TP 算出本 rank 的头数 `num_heads`、`num_kv_heads`。传给并行层的是完整大小，由并行层自己除以 N；传给 `Attention` 的是本 rank 的头数。前向时 `qkv.split` 按本 rank 的大小切开，切分点能对上，是因为 `QKVParallelLinear` 按 `[q \| k \| v]` 排布了权重 |
| **Qwen3MLP** | `MergedColumnParallelLinear` → `SiluAndMul` → `RowParallelLinear` | `SiluAndMul` 把输入从中间切成两半当 gate 和 up，本身不知道 TP 的存在；它能正确工作，依赖于 `MergedColumnParallelLinear` 让本 rank 的输出恰好是 `[gate 1/N \| up 1/N]` |
| **Qwen3DecoderLayer** | 两个 `RMSNorm` + `Qwen3Attention` + `Qwen3MLP` | pre-norm 结构，残差相加融合进下一个 norm。norm 不切：两个子模块都以 all-reduce 结束，norm 的输入在每个 rank 上都是完整的 |
| **Qwen3Model** | `VocabParallelEmbedding` + 36 个 decoder 层 + 最后的 `RMSNorm` | embedding 以 all-reduce 结束，第一层的列并行层因此能拿到完整的输入 |
| **Qwen3ForCausalLM** | `Qwen3Model` + `ParallelLMHead` | `forward` 只算到 hidden states，logits 由 `compute_logits` 单独计算（CUDA Graph 只录到 hidden states，见 [cuda-graph.md](cuda-graph.md)）。`tie_word_embeddings` 时 lm_head 直接共享 embedding 的权重，两者切词表的方式相同，所以在 TP 下依然成立（Qwen3-0.6B 共享，Qwen3-8B 不共享） |

### 3. 权重加载：checkpoint 中的名字怎样找到对应的并行层

加载机制见第五节第 3 小节。对 Qwen3 来说，checkpoint 中的名字与参数的对应关系如下：

| checkpoint 中的名字 | 对应的参数 | 调用的 loader |
|---|---|---|
| `model.embed_tokens.weight` | 同名 | `VocabParallelEmbedding` |
| `…self_attn.q_proj.weight` / `k_proj` / `v_proj` | `…self_attn.qkv_proj.weight` | `QKVParallelLinear`，`shard_id = "q"` / `"k"` / `"v"` |
| `…self_attn.o_proj.weight` | 同名 | `RowParallelLinear` |
| `…mlp.gate_proj.weight` / `up_proj` | `…mlp.gate_up_proj.weight` | `MergedColumnParallelLinear`，`shard_id = 0` / `1` |
| `…mlp.down_proj.weight` | 同名 | `RowParallelLinear` |
| `…layernorm.weight`、`…q_norm.weight` 等 | 同名 | `default_weight_loader` |
| `lm_head.weight` | 同名 | `ParallelLMHead`（继承自 embedding） |

名字的改写依靠 `Qwen3ForCausalLM.packed_modules_mapping`：loader 发现名字中含有 `q_proj`，就替换成 `qkv_proj`，并把 `"q"` 作为 `shard_id` 传给 `weight_loader`。

### 4. 小结

1. **模型代码几乎不感知 TP**：只有 `Qwen3Attention` 需要显式计算本 rank 的头数。其余地方只要选对并行层，切分、加载和通信都封装在并行层内部；
2. **配对原则**：两个子模块都是"列并行 → 按头或逐元素的操作 → 行并行"，每个子模块恰好一次 all-reduce；
3. **层与层的边界上，hidden 总是完整的**：embedding 和每个行并行层都以 all-reduce 结束，所以 norm、残差可以在每个 rank 上直接复制计算，下一个列并行层也能拿到完整的输入；
4. **首尾按词表切**：embedding 以 all-reduce 结束，lm_head 以 gather 到 rank 0 结束，两者切法相同，因此支持共享权重。
