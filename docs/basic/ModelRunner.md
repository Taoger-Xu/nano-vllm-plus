# ModelRunner 设计

[model_runner.py](../../nanovllm/engine/model_runner.py) 一共约 260 行，是整个系统中**唯一直接持有 GPU 资源的组件**。本文从五个方面展开：它的定位、多进程结构、启动流程、每步执行流程，以及设计上的取舍。

整体架构见 [overview.md](overview.md)。

---

## 1. 定位与职责

```
Scheduler（CPU，整数和列表）
   │  list[Sequence]：block_table、num_cached_tokens、num_scheduled_tokens ...
   ▼
ModelRunner  ← 把"调度决策"翻译成"GPU 张量"并执行
   │  input_ids / positions + Context(cu_seqlens, slot_mapping, block_tables ...)
   ▼
Model（Qwen3）→ logits → Sampler → list[int] token_ids
```

### 这条流程在哪里串起来

这条流程分两段串起来：**`LLMEngine.step()` 把调度层和执行层连起来**，**`ModelRunner.run()` 把执行层内部从张量到 token 的各步连起来**。最外层由 `generate()` 循环驱动。

**第一段：`LLMEngine.step()`，连接 Scheduler 和 ModelRunner**（[llm_engine.py](../../nanovllm/engine/llm_engine.py#L49-L55)）

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()                    # ① Scheduler 产出 list[Sequence]
    num_tokens = ...
    token_ids = self.model_runner.call("run", seqs, is_prefill)     # ② 交给 ModelRunner，拿回 list[int]
    self.scheduler.postprocess(seqs, token_ids, is_prefill)         # ③ token 回到 Scheduler
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

这一段对应上图的第一个箭头，并且多了一步：把采样结果交还给 Scheduler，形成闭环。

中间的 [`call`](../../nanovllm/engine/model_runner.py#L85-L89) 在 TP > 1 时，先把 `("run", seqs, is_prefill)` 通过共享内存广播给其他 rank，再在本地执行 `self.run(...)`。TP = 1 时，它就相当于直接调用 `run`。

**第二段：`ModelRunner.run()`，从 Sequence 到 token**（[model_runner.py](../../nanovllm/engine/model_runner.py#L214-L220)）

```python
def run(self, seqs, is_prefill):
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)  # ④
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)                                       # ⑤
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None             # ⑥
    reset_context()
    return token_ids
```

- **④ Sequence → 张量 + Context**：`prepare_prefill` / `prepare_decode` 构造出 `input_ids` 和 `positions`，同时在函数末尾调用 `set_context(...)`，把 `cu_seqlens`、`slot_mapping`、`block_tables` 等写进全局 Context。所以上图第二个箭头上的两类数据走的路径不同：
  - `input_ids` / `positions` 通过**函数返回值**传给 `run_model`；
  - Context 通过**全局变量**传给模型底层的 `Attention` 和 `LMHead`。
- **⑤ 张量 → logits**：`run_model` 调用 `self.model(input_ids, positions)`，再调用 `self.model.compute_logits(...)`。decode 时则改为 replay 对应档位的 CUDA Graph。
- **⑥ logits → token**：`Sampler` 采样，`.tolist()` 把结果转成 Python 的 `list[int]`。

**最外层：`generate()` 的循环**（[llm_engine.py](../../nanovllm/engine/llm_engine.py#L73)）：`while not self.is_finished(): self.step()`。

**完整调用链：**

```
LLMEngine.generate()                     循环
 └─ LLMEngine.step()                     ← 第一段：串联调度层和执行层
     ├─ Scheduler.schedule()             → list[Sequence], is_prefill
     ├─ ModelRunner.call("run", ...)     （TP 时广播）
     │   └─ ModelRunner.run()            ← 第二段：串联执行层内部
     │       ├─ prepare_prefill/decode   → input_ids, positions + set_context()
     │       ├─ run_model                → model.forward → compute_logits → logits
     │       │     └─ Attention / LMHead 内部通过 get_context() 读取元数据
     │       └─ sampler(...).tolist()    → list[int]
     └─ Scheduler.postprocess(seqs, token_ids)   token 回到调度层，形成闭环
```

### 职责

它负责四件事：

| 职责 | 方法 |
|---|---|
| 持有 GPU 资源：模型权重、KV cache、CUDA Graph | `__init__`、`allocate_kv_cache`、`capture_cudagraph` |
| 把 Sequence 翻译成张量 | `prepare_prefill`、`prepare_decode`、`prepare_block_tables`、`prepare_sample` |
| 执行前向和采样 | `run`、`run_model` |
| 多卡协同（张量并行） | `call`、`write_shm`、`read_shm`、`loop`、`exit` |

---

## 2. 多进程结构：同一个类，两种角色

张量并行（TP）为 N 时，每张卡上都有一个 `ModelRunner`。它们的代码完全相同，按 rank 分成两种角色：

```
主进程（rank 0, GPU 0）                      子进程（rank 1..N-1, GPU 1..N-1）
┌──────────────────────────┐                ┌──────────────────────────┐
│ LLMEngine                │                │ ModelRunner.__init__      │
│  ├─ Scheduler            │                │   ...初始化...             │
│  └─ ModelRunner(rank 0)  │   SharedMemory │   loop():                 │
│       call("run", ...)───┼──► 1MB 缓冲区 ──┼──►  read_shm → call(...)  │
│         ├ write_shm      │   + Event 通知  │      直到收到 "exit"       │
│         └ 本地执行 run    │                │                           │
└──────────┬───────────────┘                └────────────┬──────────────┘
           └──────────── NCCL：all_reduce / gather ───────┘
```

- **rank 0 既负责调度，也参与计算**：它先把调用广播出去，再在本地执行同一个方法（[`call`](../../nanovllm/engine/model_runner.py#L85-L89)）。不需要单独的驱动进程，也不会因此多空一张卡。
- **其他 rank 是纯执行者**：子进程的入口就是 `ModelRunner(config, rank, event)`（[llm_engine.py](../../nanovllm/engine/llm_engine.py#L25-L29)）。构造函数最后进入 `loop()`，这个循环在收到 `exit` 之前不会返回。所以对子进程来说，**构造函数就是它的整个生命周期**。
- **两条通信通道，各管一类数据**：
  - **控制面**：方法名和参数（`Sequence` 列表）都是 Python 对象，用 pickle 序列化后写入共享内存，再通过 `Event` 唤醒子进程（[`write_shm`](../../nanovllm/engine/model_runner.py#L76-L83) / [`read_shm`](../../nanovllm/engine/model_runner.py#L68-L74)）。
  - **数据面**：前向过程中的张量通信（`RowParallelLinear` 的 `all_reduce`、`LMHead` 的 `gather`）走 NCCL。
- **为什么控制面不用 `dist.broadcast_object_list`**：调度结果是 CPU 上的 Python 对象。直接在同一台机器的共享内存里交换，比走一次集合通信更轻量。`Sequence.__getstate__`（[sequence.py](../../nanovllm/engine/sequence.py)）也做了配合：decode 阶段只序列化最后一个 token，不传整个 `token_ids`，进一步减小每步传输的数据量。

---

## 3. 启动流程：顺序有讲究

[`__init__`](../../nanovllm/engine/model_runner.py#L17-L48) 中各步骤的先后顺序都有依赖，不能随意调换：

```
① dist.init_process_group(nccl) + set_device(rank)
② set_default_dtype(模型 dtype) + set_default_device("cuda")
③ 构建 Qwen3ForCausalLM → load_model 加载权重
④ warmup_model           ← 必须在分配 KV cache 之前
⑤ allocate_kv_cache      ← 必须在录制 CUDA Graph 之前
⑥ capture_cudagraph
⑦ 恢复 default device = cpu、dtype 为默认值
⑧ TP > 1：rank 0 创建 SharedMemory → barrier；其他 rank barrier → attach → loop()
```

### ② 临时修改默认 device 和 dtype

模型参数在 `nn.Parameter(torch.empty(...))` 时**直接在 GPU 上以目标精度创建**，不用先在 CPU 上建 fp32 再搬到 GPU。到第 ⑦ 步再改回 CPU，这样后面 `prepare_*` 里用 `torch.tensor(...)` 创建的张量默认在 CPU 上，方便先放进 pinned memory 再异步拷贝到 GPU。

### ④ warmup：测出激活的峰值显存

[`warmup_model`](../../nanovllm/engine/model_runner.py#L91-L101) 构造一个最大规模的 prefill（总 token 数等于 `max_num_batched_tokens`）跑一遍。目的是测出**激活值占用的峰值显存**。

此时 KV cache 还没分配，`Attention` 发现 `k_cache` 为空就会跳过写入；`Sequence` 没有 `block_table`，`prepare_prefill` 也会跳过计算 `slot_mapping`。这次 warmup 还会触发 `torch.compile` 的编译。

### ⑤ KV cache 容量规划

[`allocate_kv_cache`](../../nanovllm/engine/model_runner.py#L103-L121) 的核心公式：

```
可用于 KV 的显存 = total × gpu_memory_utilization − used − (peak − current)
                   └── 总预算 ──────────────────┘   │       └ warmup 时激活的瞬时峰值，以后每步还会出现
                                                    └ 当前已占用（权重、CUDA context 等）
num_blocks = 可用于 KV 的显存 // block_bytes
block_bytes = 2(K 和 V) × 层数 × block_size × 本卡 kv_heads × head_dim × dtype 字节数
```

分配出一个形状为 `[2, L, num_blocks, block_size, kv_heads, head_dim]` 的大张量，然后按层切片，挂到每个 `Attention` 模块的 `k_cache` / `v_cache` 属性上。

算出的 `num_blocks` 会写回 `config`，主进程随后创建 `Scheduler` 时用它初始化 `BlockManager`。**调度层的 block id 和这里物理张量的第 3 维下标是一一对应的。**

### ⑥ CUDA Graph 录制

[`capture_cudagraph`](../../nanovllm/engine/model_runner.py#L223-L257) 有几个要点：

- **固定缓冲区**：`input_ids`、`positions`、`slot_mapping`、`context_lens`、`block_tables`、`outputs` 都按最大 batch size 预分配。每张图录制时用的是这些缓冲区的 `[:bs]` 切片，图里记下的就是这些固定地址。
- **分档**：bs 取 1、2、4、8，之后每 16 一档，直到 `min(max_num_seqs, 512)`。
- **从大到小录制，共享一个显存池**：第一张图（最大的 bs）录制完后，用 `graph.pool()` 拿到它的显存池，后面的图都复用这个池，图的中间张量不会重复占用显存。
- **每档先跑一次再录制**：先在 eager 模式下跑一遍，确保 lazy 初始化、compile 等一次性开销不会被录进图里。
- **必须在 KV cache 分配之后录制**：图里会记录 `k_cache` / `v_cache` 的地址。

---

## 4. 每步执行流程

[`run`](../../nanovllm/engine/model_runner.py#L214-L220)：

```
prepare_prefill / prepare_decode    ① CPU 上构造 Python 列表 → pinned 张量 → 异步拷贝到 GPU → set_context
prepare_sample（仅 rank 0）          ② 每条序列的 temperature
run_model                           ③ 前向：eager 或 CUDA Graph
sampler（仅 rank 0）→ .tolist()      ④ 采样，并同步回 CPU
reset_context                       ⑤ 清空本步的元数据
```

### ① 输入准备：把逻辑状态翻译成物理地址

**Prefill**（[`prepare_prefill`](../../nanovllm/engine/model_runner.py#L129-L170)）

每条序列本步要算的区间是 `[start, end) = [num_cached_tokens, num_cached_tokens + num_scheduled_tokens)`。

- `input_ids` / `positions`：只包含这个区间的 token。已缓存的前缀不重复计算。
- `cu_seqlens_q` 按 `end - start` 累加，`cu_seqlens_k` 按 `end` 累加。也就是说，**Q 只包含新 token，K 包含整条序列的前缀**（prefix cache 或之前的 chunk）。
- `slot_mapping`：逐个 block 计算新 token 在物理 cache 中的写入位置。首尾 block 可能只写一部分，代码里分别处理了 `start % block_size` 和 `end - i * block_size` 这两种边界。
- `block_tables`：**只有当 `cu_seqlens_k > cu_seqlens_q` 时才构造**。这表示存在前缀，attention 要从分页 cache 中读取历史 K/V；没有前缀时，直接用本步刚算出的 K/V，效率更高。

举例：block_size = 4，某序列共 10 个 token，前 4 个命中 prefix cache，`block_table = [7, 2, 9]`：

```
start = 4, end = 10
input_ids    = token[4:10]               positions = [4..9]
seqlen_q = 6, seqlen_k = 10
slot_mapping = [2*4+0 .. 2*4+3] + [9*4+0, 9*4+1]  = [8, 9, 10, 11, 36, 37]
```

**Decode**（[`prepare_decode`](../../nanovllm/engine/model_runner.py#L172-L188)）

每条序列只处理 1 个 token：

- `input_ids` 为 `last_token`，`positions` 为 `len - 1`；
- `slot = block_table[-1] × block_size + last_block_num_tokens - 1`，即最后一个 block 中的最后一个位置；
- `context_lens` 为 `len`；
- `block_tables` 由 [`prepare_block_tables`](../../nanovllm/engine/model_runner.py#L123-L127) 把各序列的 block_table 用 -1 补齐成矩形。

所有张量都按 `pin_memory=True` → `.cuda(non_blocking=True)` 的方式创建。pinned memory 让主机到设备的拷贝可以异步进行，不阻塞 CPU。

### ③ run_model：两条执行路径

[`run_model`](../../nanovllm/engine/model_runner.py#L196-L212)：

```python
if is_prefill or enforce_eager or bs > 512:
    eager 前向                     # prefill 每步的形状变化太大，不适合用图
else:
    选择 ≥ bs 的最小档位的图
    把本步数据拷进 graph_vars 的 [:bs] 部分
    slot_mapping 先全部填 -1，context_lens 先全部清零   # 多出来的 padding 行：不写 cache，也不读历史
    graph.replay()
compute_logits(outputs[:bs])       # LMHead 放在图外执行
```

- padding 行的 `slot_mapping = -1`，`store_kvcache` 这个 kernel 遇到 -1 就直接返回，所以 padding 行不会写坏别人的 KV cache。
- `compute_logits` 放在图外执行：图只负责输出 `[bs, hidden]`，LMHead（以及 TP 下的 `gather`）在图外运行。

### ④ 采样只在 rank 0 上进行

TP 下，`LMHead` 会把各卡的 logits `gather` 到 rank 0，其他 rank 拿到的是 `None`。`temperatures` 也只在 rank 0 上准备。最后 `.tolist()` 把结果拷回 CPU，交给 `Scheduler.postprocess`。

---

## 5. 设计亮点与取舍

### 亮点

1. **职责边界清晰**：调度层和 ModelRunner 之间只通过 `Sequence` 交接，所有张量都在 ModelRunner 内部创建和销毁。
2. **SPMD + 驱动者兼任执行者**：所有 rank 运行同一份代码，执行同一个方法，只用 `rank == 0` 区分采样和广播。TP 的实现因此非常简洁。
3. **显存利用最大化**：先用 warmup 测出峰值，再把剩余显存全部分给 KV cache；多张 CUDA Graph 共享一个显存池。
4. **统一的输入抽象**：prefill、chunked prefill、prefix cache 共用同一套 `[start, end)` 加 `cu_seqlens_q/k` 的表示，不需要分别处理。

### 局限和需要注意的地方

| 问题 | 说明 |
|---|---|
| CPU 与 GPU 串行 | 每步依次执行"准备输入 → 前向 → `.tolist()` 同步 → 调度下一步"，CPU 准备输入时 GPU 处于空闲。vLLM v1 会把下一步的调度和本步的 GPU 计算重叠起来。 |
| 硬编码的端口和共享内存名 | `tcp://localhost:2333` 和 `SharedMemory(name="nanovllm")` 都是写死的，同一台机器上不能同时启动两个实例。 |
| 共享内存只有 1MB，且不检查越界 | prefill 阶段要传输每条序列的完整 `token_ids`。如果 batch 里有很多长 prompt，pickle 后的数据可能超过 1MB，而 `write_shm` 没有做大小检查。 |
| 各 rank 独立计算 block 数 | 每个子进程拿到的是 `config` 的副本，各自计算 `num_kvcache_blocks`，而 `Scheduler` 只按 rank 0 的数值分配 block。如果各卡的空闲显存不一致（比如某张卡上还有其他进程），可能出现 block id 越界。 |
| 广播的同步依赖隐式保证 | `write_shm` 不会等子进程读完。之所以不会覆盖还没读取的数据，是因为每次 `run` 中都有 NCCL 集合通信（`all_reduce`），它会让所有 rank 自然对齐。 |
| 模型类写死 | 直接实例化 `Qwen3ForCausalLM`，新增模型需要加分发逻辑，见 [overview.md](overview.md#新增一个模型需要做什么)。 |
