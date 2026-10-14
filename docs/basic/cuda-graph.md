# CUDA Graph：decode 阶段的录制与重放

本文说明 nano-vllm 如何用 CUDA Graph 加速 decode：

1. CUDA Graph 是什么，解决什么问题；
2. 一次 decode 前向涉及哪些 kernel，每个 kernel 录死了哪些参数、读写哪些显存地址；
3. 录制在什么时候、以什么方式发生，每个 decode 步如何重放；
4. 为了支持录制，代码中做了哪些设计；
5. 什么情况下不能录图；
6. 推测解码要怎样设计，才能继续使用 CUDA Graph；
7. 常见问题。

阅读前建议先看 [ContinuousBatching.md](ContinuousBatching.md) 和 [PagedAttention.md](PagedAttention.md)，本文会用到 `slot_mapping`、`block_tables`、`context_lens` 等概念。

---

## 一、CUDA Graph 是什么

### 1. 问题：decode 时 GPU 在等 CPU

PyTorch 默认是 **eager 模式**：Python 每执行一个算子，CPU 就向 GPU 发射（launch）一个 kernel。每次发射在 CPU 端都有固定开销（Python 解释器、PyTorch 分发、CUDA driver），量级是**每个几微秒**。

一次前向要发射多少个 kernel？Qwen3 每层 decoder 大约有十几个 kernel（见第二节），Qwen3-8B 有 36 层，加起来**一次前向要发射几百个 kernel**。

```
prefill（几千个 token）：每个 kernel 在 GPU 上要算很久
  CPU: |launch|launch|launch|...
  GPU:  |=====kernel1=====|=====kernel2=====|=====kernel3=====|
  → GPU 一直在忙，发射开销被掩盖

decode（每条序列 1 个 token）：每个 kernel 在 GPU 上只要几微秒
  CPU: |launch|launch|launch|launch|...
  GPU:  |k1|   |k2|   |k3|   |k4|
  → GPU 大部分时间在空等 CPU 发下一个 kernel
```

所以 decode 时，瓶颈可能根本不在 GPU，而在 **CPU 发射 kernel 的速度**。

### 2. 录一次，重放多次

CUDA Graph 把一整串 kernel 的调用**录制**下来，包括：

- 每个 kernel 是什么；
- 它的参数，包括形状、grid 大小和标量参数；
- 它读写哪些显存地址。

之后只需要一次 `graph.replay()`，GPU 就会自己把整串 kernel 跑完，CPU 端只付出一次发射的开销。

```
录制（启动时一次）：    正常跑一遍前向，CUDA 记下所有 kernel 及其参数 → 得到一个 graph
重放（每个 decode 步）：把新输入拷进录制时用的那块显存 → graph.replay() → 从同一块显存读输出
```

### 3. 核心思想：变的是数据，不变的是形状和地址

录下来的只有 GPU 上的 kernel 序列，**CPU 端的 Python 逻辑不会被录下来，重放时也不会执行**。所以一个图能被反复使用的前提是：

> 每一步之间变化的东西，全部能表达成**"固定形状、固定地址的缓冲区里的数据"**。

decode 阶段恰好满足这一点。哪个 token、第几个位置、KV 写到哪里、上下文有多长、用了哪些物理块，都只是缓冲区中的数据，形状只取决于 batch size。第二节和第三节会具体展开。

---

## 二、decode 阶段涉及哪些 kernel

### 1. 一次 decode 前向的 kernel 清单

以 bs = 4 的图为例，`self.model(input_ids, positions)` 依次发射下面这些 kernel（[qwen3.py](../../nanovllm/models/qwen3.py)）。每层的部分重复 `num_hidden_layers` 次。

| 位置 | kernel | 实现 | 录死的参数 | 读 | 写 |
|---|---|---|---|---|---|
| 开头 | embedding | `F.embedding` | 行数 = 4 | **`input_ids` 缓冲区**、词表权重 | hidden |
| 每层 | input_layernorm（加残差） | `@torch.compile` 生成的 Triton kernel | 形状 `[4, hidden]` | hidden、residual、norm 权重 | hidden、residual |
| 每层 | qkv_proj | cuBLAS GEMM | M=4、N、K | hidden、QKV 权重 | qkv |
| 每层 | q_norm / k_norm | `@torch.compile` | 形状 `[4, H, D]` | q、k | q、k |
| 每层 | rotary | `@torch.compile` | 4 个 token | **`positions` 缓冲区**、cos/sin 表 | q、k |
| 每层 | **store_kvcache** | 手写 Triton kernel | grid = 4，**KV cache 的基地址** | k、v、**`slot_mapping` 缓冲区** | **KV cache 基地址 + slot × D** |
| 每层 | **flash_attn_with_kvcache** | FlashAttention | batch=4、q_len=1、头数、`block_table` 的列数、**KV cache 的基地址** | q、**`context_lens` 缓冲区**、**`block_tables` 缓冲区**、KV cache | attention 输出 |
| 每层 | o_proj | cuBLAS GEMM（TP 时加 NCCL all_reduce） | M=4、N、K | attention 输出、权重 | hidden |
| 每层 | post_attention_layernorm（加残差） | `@torch.compile` | `[4, hidden]` | hidden、residual | hidden、residual |
| 每层 | gate_up_proj | cuBLAS GEMM | M=4、N、K | hidden、权重 | gate_up |
| 每层 | SiluAndMul | `@torch.compile` | `[4, 2·inter]` | gate_up | act |
| 每层 | down_proj | cuBLAS GEMM（TP 时加 all_reduce） | M=4、N、K | act、权重 | hidden |
| 结尾 | 最后的 norm | `@torch.compile` | `[4, hidden]` | hidden、residual | hidden |
| 结尾 | 拷贝到输出 | copy kernel | `[4, hidden]` | hidden | **`outputs` 缓冲区** |

说明：

- `split`、`view`、`flatten` 只改变张量的视图，不发射 kernel；
- `@torch.compile` 修饰的函数会被编译成一到几个融合的 Triton kernel，具体数量由编译器决定；
- 表中加粗的是**每步都会变化的数据所在的缓冲区**，它们全部来自 `graph_vars`（第三节）；
- 表中所有"中间张量"（hidden、qkv、act 等）的地址，在录制时从内存池中分配，之后固定不变（第三节第 4 小节）。

### 2. 两个关键 kernel：动态的位置被当作数据读入

只有两个 kernel 需要知道"这一步的序列是什么样的"，它们都把动态的信息当作**数据**来读，而不是当作参数：

**store_kvcache**（[attention.py:23](../../nanovllm/layers/attention.py#L23)）录死的只是 KV cache 的**基地址**。具体写到哪个位置，是 kernel 运行时从 `slot_mapping` 中读出来再计算的：

```python
slot = tl.load(slot_mapping_ptr + idx)     # 运行时读：这个 token 的 KV 写到第几个 slot
if slot == -1: return                       # 补齐的行：不写
cache_offsets = slot * D + tl.arange(0, D)
tl.store(k_cache_ptr + cache_offsets, key)
```

**flash_attn_with_kvcache**（[attention.py:72](../../nanovllm/layers/attention.py#L72)）录死的是 batch、头数和 `block_tables` 的形状。每条序列实际有多长、用了哪些物理块，是运行时从 `context_lens` 和 `block_tables` 中读出来的。长度为 0 的行什么都不读。

### 3. 不在图里的部分

图的边界停在 hidden states。[run_model](../../nanovllm/engine/model_runner.py#L198-L214) 在 `replay()` 之后，才以 eager 模式执行：

- `compute_logits`：LMHead 的 GEMM，TP 时还要把 logits gather 到 rank 0；
- 采样：`Sampler` 是单独 `@torch.compile` 的函数，最后的 `.tolist()` 需要 CPU 等待 GPU 的结果，无法录制（第五节限制 ④）。

---

## 三、如何录制

### 1. 什么时候录：引擎启动时，只录一次

录制**不是**在每个请求 prefill 之后发生的，而是在引擎启动、还没有任何请求时就完成了。之后所有请求的所有 decode 步，都只是重放这些图。

调用链：`LLM(...)` → `LLMEngine.__init__` → `ModelRunner.__init__`（[model_runner.py:31-37](../../nanovllm/engine/model_runner.py#L31-L37)）：

```python
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)     # ① 加载权重
self.sampler = Sampler()
self.warmup_model()                      # ② 用假数据跑一次最大规模的 prefill，测激活峰值
self.allocate_kv_cache()                 # ③ 用剩余显存分配 KV cache
if not self.enforce_eager:
    self.capture_cudagraph()             # ④ 录制
```

顺序是必须的：

- **④ 必须在 ③ 之后**：图中录死了 KV cache 的基地址，KV cache 必须先存在；
- **③ 必须在 ④ 之前**：KV cache 的大小根据 ② 测出的激活峰值计算，如果先录图，图的内存池会干扰这个计算。

TP > 1 时，每个 rank 在自己的进程中各自执行这个流程，各录各的图。

### 2. 固定的缓冲区：graph_vars

`capture_cudagraph`（[model_runner.py:225](../../nanovllm/engine/model_runner.py#L225)）首先一次性分配好所有输入输出缓冲区。它们的地址从此不再改变（下面的地址是虚构的，只为说明）：

```
graph_vars（按最大档位 max_bs = min(max_num_seqs, 512) 分配）
  input_ids     [512]          int64    @ 0x7f00_1000    每个 token 一份
  positions     [512]          int64    @ 0x7f00_2000    每个 token 一份
  slot_mapping  [512]          int32    @ 0x7f00_3000    每个 token 一份
  context_lens  [512]          int32    @ 0x7f00_4000    每条序列一份
  block_tables  [512, 16]      int32    @ 0x7f00_5000    每条序列一行；列数 = max_model_len / block_size
  outputs       [512, hidden]  bf16     @ 0x7f00_9000    每个 token 一行
```

**为什么大部分缓冲区是一维的？** nano-vllm 不用 HuggingFace 那种 `[batch, seq_len]` 加补齐的布局，而是把**所有序列的 token 首尾相接拼成一维**。模型中绝大部分计算（embedding、norm、GEMM、rotary、MLP）都是对每个 token 独立进行的，不关心它属于哪条序列；只有 attention 需要序列边界，而边界通过 `context_lens` / `block_tables`（decode）或 `cu_seqlens`（prefill）单独传入。decode 时每条序列恰好 1 个 token，所以 **token 数 = 序列数 = bs**，前四个缓冲区看起来形状一样，但含义不同：前三个按 token 计，`context_lens` 按序列计。`block_tables` 本来就是二维的。

进入模型后，形状变化如下：

```
input_ids [4] → embedding → [4, hidden] → qkv_proj → [4, (H+2·Hkv)·D] → view → q [4, H, D]
  → attention 内部 q.unsqueeze(1) → [4, 1, H, D]    ← 只在这里临时补出"每条序列的 token 数"这一维
  → o.flatten(1, -1) → [4, H·D] → o_proj、MLP → outputs [4, hidden]
```

### 3. 录制循环

[model_runner.py:236-250](../../nanovllm/engine/model_runner.py#L236-L250)：

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))     # 档位
self.graph_pool = None
for bs in reversed(self.graph_bs):                                    # 从大到小
    graph = torch.cuda.CUDAGraph()
    set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])        # 预热
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # 录制
    if self.graph_pool is None:
        self.graph_pool = graph.pool()                                # 后面的图共用第一个图的内存池
    self.graphs[bs] = graph
```

几个要点：

- **按档位录制**：decode 的形状只取决于 batch size，而 batch size 的范围在启动时就知道，所以可以预先穷举。默认配置下共 36 个档位；
- **传入的是缓冲区的切片**：`set_context` 中放的就是 `graph_vars` 的前 bs 行。attention 通过全局的 `Context` 读取这些张量，所以图中录下的正是这些固定地址；
- **录制前先预热一次**：第一次运行时会有懒加载的初始化（cuBLAS 工作区、Triton kernel 编译、`torch.compile` 对新形状的编译），这些操作不能出现在录制过程中；
- **用的是假数据**：缓冲区全部初始化为 0。录制只需要形状和地址，不需要真实的 token。预热时 `slot_mapping` 全为 0，会把垃圾 KV 写进 KV cache 的第 0 个 slot，但这时还没有任何请求，这个 slot 将来被分配时会被覆盖，所以无害。

### 4. 共享内存池，从大到小录制

前向中所有的**中间张量**（每层的 hidden、qkv、attention 输出、MLP 中间结果）也必须有固定地址。录制时，它们从一个专门的**内存池**中分配；录制结束后，这些地址被图"占住"，不能还给普通的 PyTorch 分配器。

如果每个图有自己的池，36 个档位的显存就要加起来。但这些图**永远不会同时运行**：每一步只 replay 其中一个，跑完后中间激活就没用了。所以它们可以共用一个池。共用之所以安全，还因为每个图唯一需要保留的结果写在 `outputs` 中，而 `outputs` 是在图外预先分配的（`outputs[:bs] = ...` 在图中只是一次拷贝），不在池里。

**为什么要从大到小录制？** 分配器在池中会优先复用已有的空闲块，前提是块足够大；块不够大时才向 GPU 申请新的显存段。用一组示意数字说明（假设 bs=512 时中间激活需要 400MB）：

```
从大到小：
  录 bs=512：池为空 → 申请 400MB；录完后中间张量释放回池
  录 bs=496：需要约 390MB → 从池里已有的 400MB 中切出来，不申请新显存
  ...
  录 bs=1：  同上
  池的总大小 ≈ 400MB（= 最大的那个图）

从小到大：
  录 bs=1：  申请 0.8MB
  录 bs=2：  需要 1.6MB，池里的块放不下 → 再申请
  录 bs=4：  再申请
  ...
  池的总大小 ≈ 所有档位之和
```

小块拼不出大块，而大块可以切出小块。先录最大的图，一开始就把池撑到峰值，后面更小的图都能在里面复用。录图发生在 KV cache 分配之后，用的是 `gpu_memory_utilization` 之外的余量，所以把图的开销压到"最大的一个图"的大小非常重要。

### 5. 每个 decode 步：拷贝数据，然后 replay

假设当前 running 中有 3 条序列（block_size = 256）：

| 序列 | 长度 | last_token | block_table | 本步位置 | 本步 slot |
|---|---|---|---|---|---|
| A | 300 | 1234 | [5, 9] | 299 | 9×256 + 43 = **2347** |
| B | 10 | 88 | [2] | 9 | 2×256 + 9 = **521** |
| C | 513 | 777 | [7, 3, 11] | 512 | 11×256 + 0 = **2816** |

slot 由 [prepare_decode](../../nanovllm/engine/model_runner.py#L174) 计算：`block_table[-1] * block_size + last_block_num_tokens - 1`。

bs = 3，向上取到档位 **4**。[run_model](../../nanovllm/engine/model_runner.py#L198-L214) 把数据拷进缓冲区：

```
input_ids[:3]        = [1234, 88, 777]          input_ids[3] 是上一步留下的旧值，无所谓
positions[:3]        = [299, 9, 512]
slot_mapping         = fill(-1)，再写 [:3] = [2347, 521, 2816]     → 第 4 行 = -1
context_lens         = 清零，再写 [:3] = [300, 10, 513]            → 第 4 行 = 0
block_tables[:3, :3] = [[5, 9, -1],
                        [2, -1, -1],
                        [7, 3, 11]]
```

然后 `graph.replay()`，**一次发射**，GPU 按录好的顺序跑完全部 kernel：

- embedding 从 `input_ids` 的地址读到 `[1234, 88, 777, 旧值]`；
- rotary 从 `positions` 的地址读到 `[299, 9, 512, …]`，各自旋转；
- store_kvcache 读到 `slot_mapping = [2347, 521, 2816, -1]`，把前三行的 K/V 写进对应的 slot，第 4 行跳过；
- flash_attn_with_kvcache 读到 `context_lens = [300, 10, 513, 0]`：A 沿着物理块 5、9 读前 300 个 KV，B 读块 2 中的前 10 个，C 读块 7、3、11 中的前 513 个，第 4 行什么都不读；
- 结果写进 `outputs`。

`replay()` 返回后，取 `outputs[:3]` 在图外计算 logits 并采样。第 4 行是补齐出来的，结果无意义，直接丢弃。

**下一步**：每条序列长度各加 1，只需要往**同样的地址**拷新数据（`positions = [300, 10, 513]`，`slot_mapping = [2348, 522, 2817]`，`context_lens = [301, 11, 514]`……），再次 replay 同一个图。如果某条序列结束、bs 变成 2，就换用 bs=2 的图；如果有新序列加入、bs 变成 5，就换用 bs=8 的图。

### 6. 一个请求的生命周期中，图是怎么用的

```
启动 ─────────────────────────────────────────────────────────────────
  加载权重 → warmup → 分配 KV cache → 录制 bs=512, 496, …, 16, 8, 4, 2, 1 的图

请求到来 ─────────────────────────────────────────────────────────────
  step 1   prefill                 eager 模式，逐个发射 kernel
  step 2   decode（bs=3）          拷贝数据 → replay bs=4 的图
  step 3   decode（bs=3）          拷贝数据 → replay bs=4 的图
  step 4   decode（bs=5）          拷贝数据 → replay bs=8 的图
  ...
  step N   decode（bs=600 > 512）  eager 模式（没有录这么大的档位）

退出 ─────────────────────────────────────────────────────────────────
  exit() 中 del self.graphs, self.graph_pool，再销毁进程组
```

---

## 四、为了支持录制，代码中做了哪些设计

| 设计 | 位置 | 解决的问题 |
|---|---|---|
| **预分配固定的输入输出缓冲区 `graph_vars`**，每步拷贝数据进去，而不是新建张量 | `capture_cudagraph`、`run_model` | 图录死的是地址，新建的张量地址会变 |
| **`Context` 中放的是 `graph_vars` 的切片** | `capture_cudagraph` 中的 `set_context` | attention 通过全局 `Context` 读取 `slot_mapping` 等张量；录制时传入切片，重放时 kernel 读的就是这些固定地址 |
| **decode 的 attention 使用 `flash_attn_with_kvcache`** | [attention.py:72](../../nanovllm/layers/attention.py#L72) | 它的动态信息全部来自张量（`cache_seqlens`、`block_table`），没有 `max_seqlen` 这类 Python 标量参数 |
| **`block_tables` 的列数固定为 `max_model_len / block_size`** | `capture_cudagraph` | 形状不随序列的实际长度变化；`run_model` 只写入实际用到的列 |
| **补齐行：`slot_mapping = -1`，`context_lens = 0`** | `run_model`；store_kvcache 中的 `if slot == -1: return` | 实际 bs 向上取整到档位后，多出来的行既不写 KV，也不读 KV |
| **按 batch size 分档录制，运行时向上取整** | `graph_bs` | decode 的 batch size 每步都在变；只需要几十个图就能覆盖 1 ~ 512 |
| **所有图共用一个内存池，从大到小录制** | `graph_pool`、`reversed(self.graph_bs)` | 图的总显存 ≈ 最大的一个图 |
| **`outputs` 在图外分配，图中只是拷贝进去** | `outputs[:bs] = self.model(...)` | 结果不在共享池中，不会被下一个图覆盖 |
| **录制前先预热一次** | 录制循环中的第一次前向 | cuBLAS 初始化、Triton 编译、`torch.compile` 编译等懒加载操作不能出现在录制过程中 |
| **KV cache 在录制之前分配** | `__init__` 中的顺序 | 图中录死了 KV cache 的基地址 |
| **图的边界停在 hidden states**，logits 和采样在图外 | `run_model` 返回前的 `compute_logits`；`run` 中的采样 | 采样最终要 `.tolist()`，需要同步，无法录制 |
| **prefill、`enforce_eager`、bs > 512 时走 eager** | `run_model` 的第一个分支 | 这些情况的形状无法预先穷举，或者没有必要录图 |
| **TP 时所有 rank 重放同一个档位的图** | rank 0 通过 `call` 广播同样的 `seqs`，各 rank 算出相同的 bs | 图中录有 NCCL all_reduce，所有 rank 必须按相同的顺序执行相同的图 |
| **退出时先删除图，再销毁进程组** | `exit()` | 图中引用了通信资源和内存池 |

---

## 五、什么情况下不能录图

### 1. 六条限制

所有限制都来自同一个事实：**录下来的只有 GPU 上的 kernel 序列，参数和地址是固定的；CPU 端的逻辑不会被录下来，重放时也不会执行。**

| 限制 | 原因 | 例子 |
|---|---|---|
| **① 形状必须固定** | kernel 的 grid 大小、张量形状都是录制时定死的 | 每步 token 数不同的 prefill；batch size 变化（所以要分档录多个图） |
| **② Python 标量参数必须固定** | 传给 kernel 的整数、浮点数被烙进了图里 | `flash_attn_varlen_func` 的 `max_seqlen_q` / `max_seqlen_k` 是 Python int，每步不同 |
| **③ 输入输出的显存地址必须固定** | 图记录的是"读写哪个地址"，而不是"读写哪个变量" | 每步新建 `torch.tensor(...)` 作为输入，图读到的还是旧地址 |
| **④ 不能有 CPU 与 GPU 的同步** | 录制过程中 CPU 不能等 GPU 的结果 | `.item()`、`.tolist()`、`.cpu()`、`print(tensor)`；输出形状取决于数据的算子，如 `nonzero`、布尔索引 `x[mask]` |
| **⑤ 不能有依赖数据的控制流** | Python 的 `if` 只在录制时执行一次，走哪个分支就被定死 | `if (x > 0).any(): ...` |
| **⑥ CPU 端的副作用不会重放** | 重放时 Python 代码根本不运行 | 在 forward 中修改 Python 变量、计数器、调用 `set_context` |

### 2. 为什么 decode 能录，prefill 不能

把每一步可能变化的东西逐项对照：

| 每步变化的东西 | decode 中怎么表达 | prefill 中怎么表达 |
|---|---|---|
| 本步的 token 数 | = 序列数，按档位补齐 → **形状只有几十种** | 1 ~ 16384 的任意值 → GEMM 的 M、所有张量的形状每步都不同（违反 ①） |
| 输入哪些 token、在什么位置 | `input_ids`、`positions` 中的**数据** | 同样是数据，但长度不定 |
| KV 写到哪里 | `slot_mapping` 中的**数据** | 同上 |
| 每条序列的上下文多长 | `context_lens` 中的**数据** | `cu_seqlens_q/k` 的长度随序列数变化，`max_seqlen_q/k` 是 **Python int**（违反 ②） |
| 用哪些物理块 | `block_tables` 中的**数据**，列数固定 | 同上 |

所以本质区别在于：decode 的所有变化都能装进"固定形状、固定地址"的缓冲区，作为数据传给 kernel；prefill 的变化体现在**形状和 Python 标量参数**上，而这两样恰好是图中录死的部分。另外，prefill 本来就是计算瓶颈，发射开销可以忽略，录图的收益也很小。

### 3. 对照 nano-vllm：哪些录了，哪些没录

| 阶段 | 是否录图 | 原因 |
|---|---|---|
| prefill | 否 | 违反 ① ②；而且是计算瓶颈，收益小 |
| decode，bs ≤ 512 | 是 | 形状只取决于 bs，按档位录制 |
| decode，bs > 512 | 否 | 没有录这么大的档位；这时 GPU 已经足够忙 |
| `enforce_eager=True` | 否 | 用于调试，或者避免录图占用额外的显存和启动时间 |
| logits、采样 | 在图外 | 采样的 `.tolist()` 违反 ④ |

---

## 六、推测解码与 CUDA Graph

推测解码的设计见 [extend/SpeculativeDecoding.md](../extend/SpeculativeDecoding.md)。这一节只讨论它和 CUDA Graph 的关系。

### 1. 为什么必须能录图

推测解码要省的是 decode 的**步数**，但每一步变复杂了：

- **验证步**：如果按 prefill 的路径来做（输入长度不定、使用 varlen 接口），就违反了 ① 和 ②，只能走 eager。省下的步数中，有一部分又被每步几百个 kernel 的发射开销吃回去；
- **草稿模型**：它的每次前向都是"小模型 + 小 batch"，GPU 上的计算量更小，**发射开销占比更高**。不录图的话，草稿模型的 K 次前向可能比大模型的一次验证还慢，加速比直接被拖垮。

### 2. 需要哪些图

| 图 | 模型 | 每条序列的 Q 长度 | 每个档位的 token 数 | 用途 | 何时录制 |
|---|---|---|---|---|---|
| `graphs`（已有） | 大模型 | 1 | bs | 普通 decode | 总是 |
| `verify_graphs` | 大模型 | K+1 | bs × (K+1) | 验证 | 开启推测解码 |
| `draft_graphs[2]` | 草稿模型 | 2 | bs × 2 | 草稿第 0 步（追赶 + 第一个草稿） | method = draft_model |
| `draft_graphs[1]` | 草稿模型 | 1 | bs | 草稿第 1..K−1 步 | method = draft_model |

所有图依然按 batch size 分档，只是每条序列的 token 数从 1 变成了固定的 q_len。现有的 `capture_cudagraph` 可以抽成 `capture(model, q_len)`，为每个 `(model, q_len)` 组合分配一组 `graph_vars`（大小为 `max_bs × q_len`）。

attention 只需要一处改动就能覆盖四种情况：`Context` 新增 `q_len` 字段，decode 分支中把 `q.unsqueeze(1)` 换成 `q.view(-1, q_len, H, D)`，算完再把输出 `view` 回 `[tokens, H, D]`。q_len = 1 时与现有行为相同。

### 3. 为了能录图，推测解码做了哪些设计

| 设计 | 满足的限制 |
|---|---|
| 验证走 decode 路径（`flash_attn_with_kvcache`），而不是 prefill 的 varlen 路径 | ②：没有 `max_seqlen` 这类 Python 标量 |
| 每条序列的验证长度统一为 K+1；n-gram 草稿不足 K 个时补齐，补齐位置 `slot_mapping = -1` | ①：形状只取决于 bs |
| 草稿模型第 0 步固定输入 2 个 token，而不是"落后时先做一次 prefill" | ① ⑤：不需要根据数据决定走哪条路径，形状永远是 `(bs, 2)` |
| 本步是否推测由整个 batch 统一决定 | ①：一个 batch 只对应一种形状的图 |
| 草稿 token 存在 ModelRunner 的局部张量中，每步拷进 `graph_vars` | ③：地址固定 |
| NgramProposer、拒绝采样、`.tolist()` 都放在图外 | ④ ⑤：CPU 上的字符串匹配、依赖数据的接受逻辑都无法录制；它们本身只是少量 kernel 或 CPU 运算，不录也没关系 |
| TP 时草稿 token 的 `dist.broadcast` 放在两次 replay 之间 | 广播的内容取决于采样结果，必须在图外进行 |
| 序列接近 `max_model_len` 时整批退回普通 decode | `block_tables` 的列数按 `max_model_len` 分配，位置 `len + K` 超出时会越界 |

### 4. 录制顺序与内存池

新增的图应该和现有的图**共用同一个内存池**，并且**先录 token 数最多的**。所有图中，单次前向 token 数最多的是 verify 图（bs × (K+1)），所以顺序是：

```
verify 图（从大到小）→ 普通 decode 图（从大到小）→ 草稿模型的两组图（从大到小）
```

这样池的大小由最大的 verify 图决定，其余所有图都在其中复用。草稿模型更小，激活也更小，同样能放进去。不同模型的图共用一个池也是安全的，因为任何时刻只有一个图在运行。

代价是启动时要多录几组图，启动时间变长；所以只在开启对应方法时才录制。

---

## 七、常见问题

**Q：录制是在每个请求完成 prefill 之后发生的吗？**
不是。录制只在引擎启动时发生一次（`ModelRunner.__init__` 中，KV cache 分配之后），那时还没有任何请求。录制需要的只是形状和地址，与请求的内容无关；而且录制很慢（每个档位要跑两遍前向），如果每个请求都录一次，开销远大于收益。

**Q：录制时用的是什么数据？**
全 0 的假数据。预热时会把垃圾 KV 写进 KV cache 的第 0 个 slot，但这时没有任何请求，将来这个 slot 被分配时会被覆盖。

**Q：`input_ids`、`positions`、`slot_mapping` 为什么是一维的？**
nano-vllm 把所有序列的 token 拼成一维，而不是 `[batch, seq_len]` 加补齐。大部分算子按 token 独立计算，只有 attention 需要序列边界，边界通过 `context_lens`、`block_tables` 或 `cu_seqlens` 单独传入。decode 时每条序列 1 个 token，所以这些缓冲区的长度就是 bs。`block_tables` 是每条序列一行，本来就是二维的。

**Q：每一步的数据都不一样，为什么能用同一个图？**
因为变化的只是缓冲区中的数据。KV 写到哪里（`slot_mapping`）、读多长的上下文（`context_lens`）、用哪些物理块（`block_tables`），都是 kernel 在运行时从这些缓冲区中**读出来**的，而不是录死在图里的参数。

**Q：实际 bs 不等于档位时，多出来的行怎么办？**
`slot_mapping` 填 -1（store_kvcache 跳过写入），`context_lens` 填 0（attention 什么都不读）。这些行的输出无意义，replay 之后只取前 bs 行。

**Q：为什么要从大到小录制？**
为了让所有图共用一个内存池，并且池只需要和最大的图一样大。大块可以切出小块，小块拼不出大块；先录最大的图，后面的图都能复用它的显存。

**Q：为什么 prefill 不录图？**
prefill 每步的 token 数可以是 1 到 16384 之间的任意值，`flash_attn_varlen_func` 的 `max_seqlen` 还是 Python 标量参数，无法预先穷举形状。而且 prefill 是计算瓶颈，发射开销可以忽略。

**Q：为什么 bs > 512 时不用图？**
没有录这么大的档位（`max_bs = min(max_num_seqs, 512)`）。batch 这么大时，每个 kernel 在 GPU 上的计算时间已经足够掩盖发射开销。

**Q：`enforce_eager=True` 有什么用？**
完全不录图，所有前向都走 eager 模式。适合调试（可以在 forward 中打断点、打印张量），或者想省下录图占用的显存和启动时间。
