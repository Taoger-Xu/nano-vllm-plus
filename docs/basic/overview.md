# nano-vllm-plus 项目总览

本文分两部分：

- **第一部分：推理引擎的设计问题**。从"处理一个 prompt"出发，逐步扩展到"同时处理很多 prompt"，列出一个 LLM 推理引擎每一步要解决的问题，以及 nano-vllm-plus 的对应方案。
- **第二部分：nano-vllm-plus 的架构**。说明它如何分层、每层负责什么、层与层之间如何协作，并跟踪一个请求在系统中的完整生命周期。

Continuous batching 的细节见 [ContinuousBatching.md](ContinuousBatching.md)，PagedAttention 与 BlockManager 的细节见 [PagedAttention.md](PagedAttention.md)，本文只概述。

---

# 第一部分：从一个 prompt 到多个 prompt

## 1. 最朴素的推理流程

拿到一个 prompt，自回归生成的最简单写法是：

```
token_ids = tokenizer.encode(prompt)
loop:
    logits = model(token_ids)           # 每次把整个序列重新算一遍
    next_id = sample(logits[-1])
    token_ids.append(next_id)
    if next_id == EOS or 长度够了: break
text = tokenizer.decode(token_ids)
```

它能跑通，但每生成一个 token，都要把前面所有 token 重新算一遍，总计算量是 O(n²)。从这里开始，推理引擎要逐个解决下面这些问题。

## 2. 单个 prompt 要解决的问题

### 2.1 避免重复计算：KV cache

Attention 里，历史 token 的 K、V 不会随着新 token 的到来而改变，所以可以缓存起来。这样推理就分成两个阶段：

| 阶段 | 输入 | 做什么 | 瓶颈 |
|---|---|---|---|
| **Prefill** | 整个 prompt（N 个 token） | 一次性算出全部 token 的 K/V，写入 cache，并产出第一个 token | 计算（compute-bound） |
| **Decode** | 上一步生成的 1 个 token | 读取 cache 中的历史 K/V，写入自己的 K/V，产出下一个 token | 显存带宽（memory-bound） |

**引擎要考虑的问题**：KV cache 放在哪里、占多少显存、怎么组织布局。

**nano-vllm-plus 的做法**：启动时预分配一整块 KV cache 张量（[model_runner.py `allocate_kv_cache`](../../nanovllm/engine/model_runner.py#L103-L121)）。每层 `Attention` 在前向时先把新的 K/V 写进 cache（Triton kernel `store_kvcache`），再进行 attention 计算（[attention.py](../../nanovllm/layers/attention.py#L59-L75)）。

### 2.2 只算需要的 logits

Prefill 阶段只需要最后一个位置的 logits 来采样下一个 token。对全部 N 个位置都做 `hidden → vocab` 的投影，是很大的浪费（词表通常有 15 万左右）。

**nano-vllm-plus 的做法**：prefill 时，`ParallelLMHead` 根据 `cu_seqlens_q` 只取每条序列最后一个 token 的 hidden state 来计算 logits（[embed_head.py](../../nanovllm/layers/embed_head.py#L57-L66)）。

### 2.3 采样与停止条件

- **采样**：nano-vllm-plus 只支持 temperature 采样，不支持 greedy（`temperature` 必须大于 0，见 [sampling_params.py](../../nanovllm/sampling_params.py)）。[Sampler](../../nanovllm/layers/sampler.py) 采用指数分布技巧（Gumbel-max 的等价形式）：`argmax(p_i / E_i)`，其中 `E_i ~ Exp(1)`。它与按 `p` 采样在数学上等价，但避免了 `torch.multinomial`，整个函数可以被 `torch.compile` 编译。
- **停止**：遇到 EOS（除非设置了 `ignore_eos`），或生成长度达到 `max_tokens`（[scheduler.py `postprocess`](../../nanovllm/engine/scheduler.py#L81-L92)）。

### 2.4 Decode 阶段的性能问题

Decode 每步只处理 1 个 token，GPU 上每个 kernel 的计算量都很小。这时**CPU 发射 kernel 的开销**和**访存次数**成了主要瓶颈。

| 问题 | 手段 | nano-vllm-plus 中的位置 |
|---|---|---|
| 一次前向要发射几百个 kernel，CPU 开销大 | **CUDA Graph**：预先录制整次前向，运行时一次性 replay | [model_runner.py `capture_cudagraph`](../../nanovllm/engine/model_runner.py#L223) |
| 小算子很多，访存次数多 | **算子融合**：用 `torch.compile` 融合 RMSNorm（含残差相加）、SiLU×Mul、RoPE、Sampler | [layernorm.py](../../nanovllm/layers/layernorm.py)、[activation.py](../../nanovllm/layers/activation.py)、[rotary_embedding.py](../../nanovllm/layers/rotary_embedding.py) |
| 多个小 GEMM | **权重合并**：Q/K/V 合并成一个 `qkv_proj`，gate/up 合并成一个 `gate_up_proj` | [linear.py](../../nanovllm/layers/linear.py)，[qwen3.py `packed_modules_mapping`](../../nanovllm/models/qwen3.py) |
| Attention 计算本身的效率 | **FlashAttention** | [attention.py](../../nanovllm/layers/attention.py) |

### 2.5 模型太大，一张卡放不下或算得太慢：张量并行

**nano-vllm-plus 的做法**：采用 Megatron 风格的张量并行（TP），在单机内最多支持 8 卡。

- `ColumnParallelLinear`（包括 QKV 和 gate_up）按输出维度切分，每张卡负责一部分 attention 头或一部分中间维度；
- `RowParallelLinear`（包括 o_proj 和 down_proj）按输入维度切分，算完后做一次 `all_reduce`；
- 词表也按卡切分：Embedding 用 mask 加 `all_reduce` 合并结果，LM Head 把各卡的 logits `gather` 到 rank 0。

这样，每个 Transformer 层只需要 2 次 `all_reduce`。

## 3. 多个 prompt 要解决的问题

单条请求的 decode 阶段几乎完全受显存带宽限制，GPU 算力大量空闲。把多条请求放在一起跑，读一次权重可以服务更多 token，吞吐会大幅提升。但"一起跑"会引出一连串新问题：

```
想要把多条请求放在一起跑（batching）
 │
 ├─ 静态 batch：短序列结束后槽位空转、新请求要等整批结束
 │     └─► continuous batching：每个 step 重新组 batch，序列随进随出
 │            │
 │            ├─ 序列长度各不相同、每步都在变 ─► 变长 attention（扁平 batch，去掉 padding）
 │            ├─ 输出长度未知，显存频繁分配和释放 ─► PagedAttention（按 block 分配 KV）
 │            ├─ 按需分配加上乐观接纳，显存可能中途耗尽 ─► 抢占（recompute）
 │            ├─ 每步跑谁？prefill 和 decode 谁优先？ ─► 调度策略 + token 预算
 │            ├─ 超长 prompt 一次 prefill 会独占一个 step ─► chunked prefill
 │            └─ decode 的 batch size 每步都在变 ─► CUDA Graph 按 batch size 分档
 │
 ├─ 很多请求共享相同前缀（system prompt、few-shot 示例） ─► prefix caching
 └─ KV cache 能用多少显存？ ─► 先跑一次最大规模的 warmup 测出峰值，剩余显存全部分给 KV cache
```

逐项对应到 nano-vllm-plus：

| 问题 | 方案 | 代码位置 |
|---|---|---|
| 组 batch 的粒度 | 每个 step 调度一次（iteration-level） | [llm_engine.py `step`](../../nanovllm/engine/llm_engine.py#L49-L55) |
| 序列长度不一 | 拼成一维 batch，配合 `cu_seqlens`；prefill 用 `flash_attn_varlen_func` | [model_runner.py `prepare_prefill`](../../nanovllm/engine/model_runner.py#L129-L170) |
| decode 时各序列上下文长度不同 | `context_lens` + `block_tables`，调用 `flash_attn_with_kvcache` | [model_runner.py `prepare_decode`](../../nanovllm/engine/model_runner.py#L172-L188) |
| KV 显存管理 | 每个 block 256 个 token，用 block table 做逻辑块到物理块的映射 | [block_manager.py](../../nanovllm/engine/block_manager.py) |
| 显存耗尽 | 踢掉最晚加入的序列，释放它的 block，之后重新 prefill | [scheduler.py `preempt`](../../nanovllm/engine/scheduler.py#L75) |
| 调度策略 | prefill 优先，不混批；受 `max_num_seqs` 和 `max_num_batched_tokens` 两个预算约束 | [scheduler.py `schedule`](../../nanovllm/engine/scheduler.py#L25-L73) |
| 超长 prompt | chunked prefill（只允许 batch 中第一条序列被切块） | [scheduler.py](../../nanovllm/engine/scheduler.py#L42) |
| batch size 变化 | 为 bs = 1, 2, 4, 8, 16, 32, …, 512 各录制一张 CUDA Graph | [model_runner.py](../../nanovllm/engine/model_runner.py#L234) |
| 公共前缀 | 对 block 做链式 xxhash，配合引用计数共享 block | [block_manager.py `can_allocate` / `hash_blocks`](../../nanovllm/engine/block_manager.py#L58) |
| KV 容量规划 | 根据 warmup 测得的峰值显存，计算 `num_kvcache_blocks` | [model_runner.py `warmup_model`](../../nanovllm/engine/model_runner.py#L91) |

## 4. 设计问题清单（小结）

设计一个推理引擎，可以按下面五个维度逐项检查：

1. **计算正确性**：KV cache、位置编码（`positions`）、causal mask、采样、停止条件。
2. **单请求性能**：只算必要的 logits，算子融合，CUDA Graph，FlashAttention，权重合并。
3. **多请求吞吐**：continuous batching，去 padding，PagedAttention，调度策略和预算。
4. **显存**：KV 容量规划，按需分配，抢占，prefix cache 复用。
5. **扩展性**：张量并行，多进程间的通信，权重的切分与加载。

nano-vllm-plus 在约 1200 行代码里覆盖了这五个维度的核心做法。它**没有**覆盖的部分见文末的局限性一节。

---

# 第二部分：nano-vllm-plus 的架构

## 1. 分层总览

```
┌──────────────────────────────────────────────────────────────────┐
│ ① 接口层     LLM / SamplingParams                                 │  llm.py, sampling_params.py
│              generate(prompts, sampling_params) -> [{"text",...}] │
├──────────────────────────────────────────────────────────────────┤
│ ② 引擎层     LLMEngine                                            │  engine/llm_engine.py
│              tokenizer · step 主循环 · 启动 TP worker 进程         │
├──────────────────────────────┬───────────────────────────────────┤
│ ③ 调度层（CPU，纯逻辑）       │ ④ 执行层（GPU 资源）               │
│   Scheduler                  │   ModelRunner                     │  engine/scheduler.py
│   Sequence                   │   KV cache 分配 · 准备输入         │  engine/sequence.py
│   BlockManager               │   CUDA Graph · 采样 · TP 通信      │  engine/block_manager.py
│   "跑谁、跑多少、KV 放哪"     │   "把决策变成张量并执行"           │  engine/model_runner.py
├──────────────────────────────┴───────────────────────────────────┤
│ ⑤ 模型层     Qwen3ForCausalLM → Qwen3Model → DecoderLayer × N     │  models/qwen3.py
│              Qwen3Attention / Qwen3MLP                            │
├──────────────────────────────────────────────────────────────────┤
│ ⑥ 算子层     Attention(KV 写入 + FlashAttn) · Parallel Linear     │  layers/*.py
│              VocabParallelEmbedding / LMHead · RMSNorm · RoPE     │
│              SiluAndMul · Sampler                                 │
├──────────────────────────────────────────────────────────────────┤
│ 横切模块     Config · Context（跨层传元数据）· loader（权重加载）  │  config.py, utils/
└──────────────────────────────────────────────────────────────────┘
```

### 分层的几条核心原则

1. **调度层只管 CPU 上的逻辑状态，执行层只管 GPU 资源。** `Scheduler` 和 `BlockManager` 里只有 Python 列表和整数，例如 block id、token 数，不碰任何张量。`ModelRunner` 则只负责把这些整数翻译成 GPU 张量（`slot_mapping`、`block_tables` 等）并执行前向。两者通过 `Sequence` 对象交接。这样调度逻辑可以单独理解和测试，执行层也不需要关心调度策略。
2. **模型代码不感知 batch 结构。** `Qwen3` 的网络结构和层的写法与 HuggingFace 版基本一致，`forward` 只接收 `input_ids` 和 `positions`。batch 的边界、KV 写入位置、block table 等元数据，通过全局 [Context](../../nanovllm/utils/context.py) 直接传给最底层的 `Attention` 和 `LMHead`，中间各层不需要修改函数签名。详见下文的[模型层与 Context](#模型层与-context为什么模型不感知-batch-结构)一节。
3. **并行策略封装在算子内部。** 张量并行的切分与通信都在 `ColumnParallelLinear`、`RowParallelLinear`、`VocabParallelEmbedding` 这些算子里完成，模型层看不到 `all_reduce`。
4. **权重怎么切分由算子自己决定。** 每个参数都挂着自己的 `weight_loader`。[loader.py](../../nanovllm/utils/loader.py) 只负责遍历 safetensors 文件，通过 `packed_modules_mapping` 把 `q_proj`/`k_proj`/`v_proj` 映射到合并后的 `qkv_proj` 的对应分片，具体如何切分由参数自己的 `weight_loader` 完成。

## 2. 各层职责详解

### ① 接口层

- [`LLM`](../../nanovllm/llm.py) 只是 `LLMEngine` 的别名，API 和 vLLM 保持一致。
- [`SamplingParams`](../../nanovllm/sampling_params.py) 有三个字段：`temperature`、`max_tokens`、`ignore_eos`。`generate` 可以给每条请求传入不同的参数。
- 只支持**离线批量推理**：一次性提交所有 prompt，阻塞到全部完成后返回。没有在线服务和流式输出。

### ② 引擎层：LLMEngine

[llm_engine.py](../../nanovllm/engine/llm_engine.py) 负责把各个组件串起来：

- **初始化**：构建 `Config`；为 rank 1 到 N-1 各 `spawn` 一个 `ModelRunner` 进程；在本进程中创建 rank 0 的 `ModelRunner`；加载 tokenizer；最后创建 `Scheduler`。
  - 顺序很重要：`Scheduler` 需要 `config.num_kvcache_blocks`，而这个值是 rank 0 的 `ModelRunner` 在测量显存之后才写入 config 的。
- **`add_request`**：把 prompt 编码成 token，包装成 `Sequence`，放进调度器的 `waiting` 队列。
- **`step`**：依次调用调度、执行、后处理，返回本步结束的序列。
- **`generate`**：循环调用 `step` 直到全部完成，同时统计 prefill 和 decode 的吞吐，最后按提交顺序 decode 成文本返回。

### ③ 调度层：Scheduler + Sequence + BlockManager

调度策略的设计取舍见 [Scheduler.md](Scheduler.md)，块管理的使用场景见 [block.md](block.md)。

| 组件 | 职责 |
|---|---|
| [`Sequence`](../../nanovllm/engine/sequence.py) | 一条请求的全部状态：`token_ids`、`status`（WAITING/RUNNING/FINISHED）、`block_table`、`num_cached_tokens`（已写入 KV cache 的 token 数）、`num_scheduled_tokens`（本步要算的 token 数）、采样参数。实现了自定义的 `__getstate__`：decode 阶段只序列化最后一个 token，减少跨进程传输的数据量。详见 [Sequence.md](Sequence.md)。 |
| [`BlockManager`](../../nanovllm/engine/block_manager.py) | 管理 KV block 的分配与回收：空闲队列、引用计数、prefix cache 的 hash 表。主要接口有 `can_allocate`/`allocate`（prefill）、`can_append`/`may_append`（decode）、`deallocate`（结束或被抢占）、`hash_blocks`（为写满的 block 登记 hash）。 |
| [`Scheduler`](../../nanovllm/engine/scheduler.py) | 维护 `waiting` 和 `running` 两个队列。`schedule()` 决定本步跑哪些序列、各跑多少 token，并返回本步是 prefill 还是 decode；`postprocess()` 追加采样结果、判断是否结束、回收资源。 |

### ④ 执行层：ModelRunner

[model_runner.py](../../nanovllm/engine/model_runner.py) 是 GPU 资源的唯一持有者。每个 TP rank 各有一个实例。更详细的设计分析见 [ModelRunner.md](ModelRunner.md)。

**启动阶段**（`__init__`）：

```
init_process_group(nccl) → 构建模型 → load_model 加载权重
  → warmup_model：用最大规模的 prefill 跑一次，测出激活峰值
  → allocate_kv_cache：(显存总量 × 利用率 − 已用 − 峰值) / 每个 block 的字节数 = block 数
                       把大张量按层切开，挂到每个 Attention 的 k_cache / v_cache 上
  → capture_cudagraph：为 decode 的各档 batch size 录制图，共享同一个显存池
  → TP > 1 时：建立共享内存；rank > 0 的进程进入 loop() 等待指令
```

**每步执行**（`run`）：

```
prepare_prefill / prepare_decode   把 Sequence 列表翻译成张量，并写入 Context
        → run_model                prefill 走 eager；decode 走 CUDA Graph replay（bs > 512 时回退到 eager）
        → Sampler                  只在 rank 0 上执行
        → reset_context
```

**TP 通信**：rank 0 通过 `call(method, *args)` 把方法名和参数用 pickle 序列化，写入 1MB 的 `SharedMemory`，再用 `Event` 通知其他 rank。其他 rank 在 `loop()` 里读出指令并执行同一个方法。这样所有 rank 用相同的输入同步执行前向；前向中的数据交换由 NCCL 的 `all_reduce`/`gather` 完成。**调度只在 rank 0 上进行**，其他 rank 只负责执行。

### ⑤ 模型层：Qwen3

[qwen3.py](../../nanovllm/models/qwen3.py) 的结构是标准的 decoder-only Transformer：

```
Qwen3ForCausalLM
 ├─ Qwen3Model
 │   ├─ VocabParallelEmbedding
 │   ├─ Qwen3DecoderLayer × N
 │   │   ├─ RMSNorm（与残差相加融合）
 │   │   ├─ Qwen3Attention: qkv_proj → q_norm/k_norm → RoPE → Attention → o_proj
 │   │   ├─ RMSNorm（与残差相加融合）
 │   │   └─ Qwen3MLP: gate_up_proj → SiluAndMul → down_proj
 │   └─ RMSNorm
 └─ ParallelLMHead（compute_logits；词表与 embedding 共享时复用 embedding 的权重）
```

残差连接以 `(hidden_states, residual)` 二元组的形式在层间传递，由下一个 RMSNorm 在内部完成"先相加、再归一化"，省掉一次单独的 add kernel。

#### 与 HuggingFace 实现的区别

结构和层的写法相近，但接口和权重布局不同：

| | HuggingFace | nano-vllm-plus |
|---|---|---|
| 输入形状 | `[batch, seq_len]`，需要 padding | 一维 `[总 token 数]`，多条序列首尾相接 |
| 额外输入 | `attention_mask`、`past_key_values` | 无，元数据从 Context 读取，KV cache 挂在 `Attention` 模块上 |
| 权重布局 | `q_proj`/`k_proj`/`v_proj`、`gate_proj`/`up_proj` 分开 | 合并为 `qkv_proj`、`gate_up_proj`，并按 TP 切分 |

#### 模型层与 Context：为什么模型不感知 batch 结构

**模型看到的输入。** `ModelRunner` 把一个 batch 里所有序列的 token 拼成一维：

```
序列 A：a1 a2 a3        序列 B：b1 b2
input_ids = [a1, a2, a3, b1, b2]        形状 [5]
positions = [0,  1,  2,  0,  1]         形状 [5]
```

进入模型后，hidden states 的形状是 `[总 token 数, hidden_size]`，没有 batch 维度。模型本身不知道 a3 和 b1 之间是两条序列的分界。

**大部分层不需要知道分界。** 逐个算子看：

| 算子 | 计算方式 | 需要知道序列边界吗 |
|---|---|---|
| Embedding | 每个 token 单独查表 | 不需要 |
| RMSNorm | 每个 token 单独归一化 | 不需要 |
| qkv_proj / o_proj / MLP | 每个 token 单独做矩阵乘 | 不需要 |
| RoPE | 每个 token 按自己的 position 旋转 | 不需要（`positions` 已显式传入） |
| **Attention** | token 之间互相"看" | **需要**：a3 不能看到 b1；还要知道 K/V 写到 cache 的哪里、历史 K/V 从哪里读 |
| **LMHead** | prefill 时只算每条序列最后一个位置 | **需要**：要知道 a3、b2 分别是各自序列的最后一个 token |

Transformer 里绝大多数计算都是逐 token 独立的，把 5 个 token 当作 5 行矩阵直接计算，结果完全正确。只有 Attention 会在 token 之间混合信息，所以只有它（以及 LMHead）需要 batch 元数据。

**元数据如何送达。** 如果用参数一层层往下传，中间每一层都要多出它自己用不到的参数：

```python
# 假设不用 Context
Qwen3ForCausalLM.forward(input_ids, positions, attn_metadata, kv_caches)
  └─ Qwen3Model.forward(input_ids, positions, attn_metadata, kv_caches)
       └─ DecoderLayer.forward(positions, h, residual, attn_metadata, kv_cache[i])
            └─ Qwen3Attention.forward(positions, h, attn_metadata, kv_cache)
                 └─ Attention.forward(q, k, v, attn_metadata, kv_cache)   ← 只有这里真正用到
```

nano-vllm-plus 用全局 Context 绕过中间层：

```python
# 执行层：prepare_prefill / prepare_decode
set_context(is_prefill, cu_seqlens_q, ..., slot_mapping, context_lens, block_tables)

# 模型层：签名保持干净
Qwen3ForCausalLM.forward(input_ids, positions)
  └─ ... └─ Attention.forward(q, k, v)
                context = get_context()                         ← 在这里直接取
                store_kvcache(..., context.slot_mapping)
                flash_attn_varlen_func(..., cu_seqlens_q=context.cu_seqlens_q, ...)

ParallelLMHead.forward(x):
    context = get_context()
    if context.is_prefill:
        x = x[context.cu_seqlens_q[1:] - 1]                    ← 只取每条序列最后一个位置
```

KV cache 也不通过参数传递：`allocate_kv_cache` 启动时直接把 cache 张量挂到每个 `Attention` 模块的 `k_cache` / `v_cache` 属性上。

对应代码：写入在 [model_runner.py `prepare_prefill`](../../nanovllm/engine/model_runner.py#L169) / [`prepare_decode`](../../nanovllm/engine/model_runner.py#L187)，读取在 [attention.py](../../nanovllm/layers/attention.py#L60) 和 [embed_head.py](../../nanovllm/layers/embed_head.py)，每步结束由 [`run`](../../nanovllm/engine/model_runner.py#L214) 中的 `reset_context()` 清空。

**好处：**

1. 模型文件只描述网络结构，没有调度、分页、batch 的概念。
2. 推理机制与模型解耦：修改 KV cache 布局、改换 attention 后端、增加元数据字段，只需改 `Context`、`ModelRunner`、`Attention`，模型文件不用动。
3. 新模型用 `layers/` 里的组件搭出结构，continuous batching、PagedAttention、prefix cache 就自动生效。

**代价：**

1. 全局隐式状态：读模型代码时看不出 `Attention` 依赖外部先调用 `set_context`；一个进程同一时间只能执行一次前向。
2. 与 CUDA Graph 的配合：Graph 录制的是固定显存地址，decode 时 Context 里的张量必须是固定缓冲区 `graph_vars` 的切片，replay 前要把新数据拷进去（[model_runner.py `run_model`](../../nanovllm/engine/model_runner.py#L197-L212)），不能每步新建张量。

#### 新增一个模型需要做什么

模型内部用 `layers/` 的组件搭建即可，但目前还有两处需要处理：

1. **模型类的选择**：[model_runner.py](../../nanovllm/engine/model_runner.py#L12) 直接 import 并实例化 `Qwen3ForCausalLM`，是写死的。新增模型时，需要在这里根据 `hf_config`（例如 `architectures` 字段）分发到对应的模型类。
2. **权重映射**：在模型类上定义 `packed_modules_mapping`，把 checkpoint 中分开的 `q_proj`/`k_proj`/`v_proj`、`gate_proj`/`up_proj` 映射到合并后参数的对应分片，[loader.py](../../nanovllm/utils/loader.py) 据此调用各参数的 `weight_loader`。

### ⑥ 算子层

| 文件 | 内容 |
|---|---|
| [attention.py](../../nanovllm/layers/attention.py) | `store_kvcache`（Triton kernel，按 `slot_mapping` 把 K/V 写入分页 cache，遇到 -1 跳过）；prefill 调用 `flash_attn_varlen_func`（有 prefix cache 时传入 `block_table`，直接从 cache 读 K/V），decode 调用 `flash_attn_with_kvcache` |
| [linear.py](../../nanovllm/layers/linear.py) | `Replicated`、`ColumnParallel`、`MergedColumnParallel`、`QKVParallel`、`RowParallel` 五种线性层，各自实现 `weight_loader` |
| [embed_head.py](../../nanovllm/layers/embed_head.py) | 按词表切分的 Embedding 和 LMHead；prefill 时只取每条序列最后一个位置计算 logits |
| [layernorm.py](../../nanovllm/layers/layernorm.py) | RMSNorm，以及融合了残差相加的版本（`torch.compile`） |
| [rotary_embedding.py](../../nanovllm/layers/rotary_embedding.py) | 预先算好 cos/sin 表，按 `positions` 查表；用 `lru_cache` 让所有层共享同一个实例 |
| [activation.py](../../nanovllm/layers/activation.py) | SiLU(gate) × up |
| [sampler.py](../../nanovllm/layers/sampler.py) | temperature 缩放加指数分布采样 |

### 横切模块

- [Config](../../nanovllm/config.py)：模型路径、`max_num_batched_tokens`、`max_num_seqs`、`max_model_len`、`gpu_memory_utilization`、`tensor_parallel_size`、`enforce_eager`、`kvcache_block_size`。其中 `num_kvcache_blocks` 和 `eos` 在运行时才写入。
- [Context](../../nanovllm/utils/context.py)：一个全局 dataclass，保存本步的 `is_prefill`、`cu_seqlens_q/k`、`max_seqlen_q/k`、`slot_mapping`、`context_lens`、`block_tables`。在执行层写入，在算子层读取。
- [loader](../../nanovllm/utils/loader.py)：遍历 safetensors 文件，把权重分发给各参数的 `weight_loader`。

## 3. 一个请求的完整生命周期

以 [example.py](../../example.py) 中的 `llm.generate(["introduce yourself", ...])` 为例：

```
[接口层] generate(prompts, sp)
   │
[引擎层] add_request: tokenizer.encode → Sequence(status=WAITING) → scheduler.waiting
   │
   ▼ ─────────────────── step 循环 ───────────────────
[调度层] schedule()
   │   waiting 中有序列 → 进入 prefill 分支
   │   BlockManager.can_allocate: 计算命中多少 prefix cache 块，检查空闲块是否够用
   │   BlockManager.allocate: 填好 seq.block_table，设置 num_cached_tokens（命中的部分）
   │   根据 token 预算设置 num_scheduled_tokens（可能被切块）
   │   prompt 全部调度完 → status=RUNNING，移入 running
   │
[执行层] ModelRunner.run(seqs, is_prefill=True)
   │   prepare_prefill: 把每条序列 [num_cached_tokens, +num_scheduled_tokens) 区间的
   │                    token 拼成一维 input_ids / positions
   │                    计算 cu_seqlens_q/k，以及 slot_mapping（每个 token 写入物理 cache 的哪个位置）
   │                    → set_context(...)
   │   TP > 1 时：先通过共享内存把调用广播给其他 rank
   │
[模型层/算子层] 每一层：
   │   qkv_proj → norm → RoPE(positions)
   │   Attention: store_kvcache(按 slot_mapping 写入) → flash_attn_varlen_func
   │   o_proj(all_reduce) → MLP(all_reduce)
   │   LMHead: 只取每条序列最后一个位置 → logits（gather 到 rank 0）
   │   Sampler → 每条序列得到一个 token_id
   │
[调度层] postprocess()
   │   hash_blocks: 为本步写满的 block 登记 hash，供后续请求复用前缀
   │   num_cached_tokens += num_scheduled_tokens
   │   prefill 还没算完（被切块）→ 不追加 token，下一步继续 prefill
   │   否则 append_token
   │
   ▼ 之后的 step：waiting 为空 → 进入 decode 分支
[调度层] 每条 running 序列 num_scheduled_tokens=1；跨入新 block 时 may_append 分配新块
   │      空闲块不够 → preempt 最晚加入的序列（释放它的块，放回 waiting 队头）
[执行层] prepare_decode: input_ids=每条序列的最后一个 token，context_lens，block_tables
   │      → 按 batch size 选择对应档位的 CUDA Graph replay
[算子层] Attention: store_kvcache → flash_attn_with_kvcache(按 block_table 读取历史 K/V)
[调度层] postprocess: 遇到 EOS 或达到 max_tokens → FINISHED，deallocate 释放块，移出 running
   │                 释放的块下一步就能被新序列使用
   ▼ ───────────────────────────────────────────────
[引擎层] 所有序列都完成 → 按 seq_id 排序 → tokenizer.decode → 返回 [{"text", "token_ids"}]
```

## 4. 关键数据结构之间的关系

```
Sequence.block_table = [7, 2, 9]       ← 逻辑块 0、1、2 分别对应物理块 7、2、9
          │
          ▼  BlockManager.blocks[7] = Block(ref_count, hash, token_ids)
          │
          ▼  ModelRunner.kv_cache[2, num_layers, num_blocks, block_size, kv_heads, head_dim]
                                       └─ 每层的 Attention.k_cache / v_cache 是它的一个视图

第 i 个 token 写入的物理位置 slot = block_table[i // block_size] * block_size + i % block_size
```

- 调度层只处理 `block_table` 中的整数；
- 执行层把它转换成 `slot_mapping`（每个 token 写到哪里）和 `block_tables`（读取时去哪里找）；
- 算子层按这两个索引访问真正的显存。

## 5. 局限性（相比完整的 vLLM）

- 只支持 Qwen3 一种模型架构；
- 只支持离线批量推理，没有在线服务、流式输出和异步请求；
- 采样只有 temperature，没有 top-k、top-p、各种 penalty，也不支持 greedy；
- prefill 和 decode 不在同一个 step 中混合，新请求进来时 decode 会停顿一步；
- chunked prefill 只允许 batch 中第一条序列被切块；
- 抢占只支持 recompute，不支持换出到 CPU；
- 张量并行只在单机内（最多 8 卡），没有流水线并行，也不支持量化。

这些取舍让核心机制保持清晰、可读，同时在小模型的离线吞吐上接近 vLLM（见 README 的 benchmark）。
