# 两个 Attention：Qwen3Attention 与 Attention

nano-vllm 中有两个名字里带 Attention 的类，容易混淆：

- [qwen3.py](../../nanovllm/models/qwen3.py) 中的 `Qwen3Attention`；
- [attention.py](../../nanovllm/layers/attention.py) 中的 `Attention`。

两者是**包含关系**：`Qwen3Attention` 是模型中完整的"注意力子层"，`Attention` 只是它内部负责"算注意力 + 管理 KV cache"的那一个核心部件。

```python
# qwen3.py
class Qwen3Attention(nn.Module):
    def __init__(...):
        self.qkv_proj = QKVParallelLinear(...)
        self.q_norm / self.k_norm = RMSNorm(...)
        self.rotary_emb = get_rope(...)
        self.attn = Attention(...)          # ← 就是 layers/attention.py 中的 Attention
        self.o_proj = RowParallelLinear(...)
```

---

## 一、一次前向中各自负责哪一段

```
hidden [T, hidden]
  │
  │  ┌──────────── Qwen3Attention（qwen3.py）────────────────────────────┐
  │  │ qkv_proj           投影出 q、k、v                                  │
  │  │ split + view       拆成 [T, H, D] / [T, Hkv, D]                    │
  │  │ q_norm / k_norm    Qwen3 特有的 q/k 归一化                          │
  │  │ rotary_emb         位置编码                                        │
  │  │   ┌──────── Attention（layers/attention.py）─────────────────┐     │
  │  │   │ store_kvcache   把本步的 k、v 写进分页 KV cache            │     │
  │  │   │ flash-attn      softmax(QKᵀ)V：prefill / decode 两条路径   │     │
  │  │   └──────────────────────────────────────────────────────────┘     │
  │  │ o_proj             输出投影（TP 时 all-reduce）                     │
  │  └────────────────────────────────────────────────────────────────────┘
  ▼
hidden [T, hidden]
```

## 二、对比

| | `Qwen3Attention` | `Attention` |
|---|---|---|
| 层次 | **模型层**：描述"Qwen3 的注意力子层长什么样" | **算子层**：描述"注意力在这个推理引擎里怎样执行" |
| 是否有权重 | 有：qkv_proj、o_proj、q_norm、k_norm | **没有**。只有 `k_cache` / `v_cache` 两个引用 |
| 输入 | hidden states `[T, hidden]` | 已经投影、加好位置编码的 q `[T, H, D]`、k / v `[T, Hkv, D]` |
| 输出 | hidden states `[T, hidden]` | 注意力结果 `[T, H, D]` |
| 是否和模型有关 | 有关：Qwen3 有 q/k norm、没有 qkv bias，换成 Llama 就要改 | **无关**：任何标准的多头 / GQA 注意力都可以直接复用 |
| 是否和引擎有关 | 基本无关，只是调用各种层 | **强相关**：读取全局 `Context`（`is_prefill`、`slot_mapping`、`cu_seqlens`、`block_tables`、`context_lens`），按 prefill / decode 选择不同的 flash-attn 接口 |
| KV cache | 不知道 KV cache 的存在 | 负责写入（`store_kvcache` 这个 Triton kernel）和读取（通过 `block_table` 读分页 cache） |
| TP | 计算本 rank 的头数，选用并行线性层（见 [TP.md](TP.md)） | 不感知 TP，只按传进来的头数（本 rank 的头）计算 |

## 三、Attention 内部做了什么

[attention.py:59](../../nanovllm/layers/attention.py#L59) 的 `forward` 只有三步：

```python
context = get_context()
store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)       # ① 写 KV cache（warmup 时 cache 为空，跳过）
if context.is_prefill:
    if context.block_tables is not None:                           # ② 有前缀命中 / chunked prefill：K/V 改从 cache 读
        k, v = k_cache, v_cache
    o = flash_attn_varlen_func(q, k, v, cu_seqlens_q=…, cu_seqlens_k=…, block_table=…)
else:
    o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,  # ③ decode：每条序列 1 个 query，K/V 全部从 cache 读
                                cache_seqlens=context.context_lens, block_table=context.block_tables)
```

- `slot_mapping`、`block_tables` 等元数据的含义见 [PagedAttention.md](PagedAttention.md)；
- prefill 时为什么只有命中前缀才走 cache 路径，见 [block.md](block.md) 中 chunked prefill 的部分；
- decode 路径为什么能录 CUDA Graph，见 [cuda-graph.md](cuda-graph.md)。

### KV cache 是怎样绑定进来的

`Attention` 的 `k_cache` / `v_cache` 在构造时是空张量。等 [model_runner.py](../../nanovllm/engine/model_runner.py) 的 `allocate_kv_cache` 分配好整块 KV cache 后，再按层号逐个绑定进来：

```python
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

ModelRunner 正是通过"有没有 `k_cache` 属性"来找到每一层的 `Attention`，所以 `Qwen3Attention` 不需要参与这件事。warmup 发生在分配 KV cache 之前，这时 cache 还是空张量，`forward` 中的 `if k_cache.numel() and v_cache.numel()` 会跳过写入。

## 四、为什么要分成两层

1. **模型和引擎解耦**：分页 KV cache、prefill / decode 两种路径、`Context` 中的各种元数据，都是推理引擎的概念，全部封装在 `Attention` 里。模型文件只需要描述网络结构，和 HuggingFace 的写法几乎一样；
2. **复用**：以后要支持 Llama、Qwen2 等模型时，只需要写新的 `XxxAttention`（投影、norm、位置编码可能不同），核心的 `Attention` 直接复用；
3. **修改集中**：像推测解码那样改变 attention 的执行方式（`q_len = K+1`，见 [SpeculativeDecoding.md](../extend/SpeculativeDecoding.md)）时，只需要改 `Attention` 一处，所有模型都能受益；
4. **配合 CUDA Graph 和 TP**：`Attention` 的所有动态信息都来自 `Context` 中的张量，满足录图的要求；它也不感知 TP，只按本 rank 的头数计算，切头的工作由外层的 `Qwen3Attention` 和并行线性层完成。

vLLM 也是同样的分层：每个模型有自己的 `XxxAttention`，内部统一调用 `vllm.attention.Attention`。
