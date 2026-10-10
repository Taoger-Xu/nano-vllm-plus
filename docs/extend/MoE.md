# 扩展一：让 nano-vllm 支持 MoE 模型

本文讨论如何在当前 nano-vllm 的架构上支持 MoE（Mixture of Experts）模型，例如 Qwen3-MoE。内容分三部分：

1. MoE 给推理引擎带来了哪些新问题；
2. 参考项目 [nano-vLLM-MS2](https://github.com/banfeb/nano-vLLM-MS2)（下文简称 **MS2**）是怎么实现的；
3. 从架构角度，在当前项目中落地时应该怎样设计、哪些地方需要优化。

阅读前建议先看 [basic/overview.md](../basic/overview.md) 中的分层架构，本文会反复用到其中的"模型层 / 算子层 / 执行层 / 调度层"等概念。

> 说明：MS2 基于较早版本的 nano-vllm（chunked prefill 重构之前），本文提到它的实现时，只借鉴设计思路；落地方案以当前项目的代码为准。对 MS2 的分析来自阅读源码，没有在 GPU 上实际运行。

---

## 一、MoE 是什么，对推理引擎意味着什么

### 1. 结构上的变化

Dense 模型的每个 decoder 层有一个 FFN（MLP）。MoE 模型把它换成：

```
                    ┌─► expert 0 (FFN) ─┐
hidden ─► router ───┼─► expert 1 (FFN) ─┼─► 按路由权重加权求和 ─► output
 (每个 token)  │     ├─►      ...        │
               │     └─► expert E-1     ─┘
               └ 每个 token 只选 top-k 个 expert（例如 128 选 8）
```

- **router（gate）**：一个 `[hidden, E]` 的线性层，算出每个 token 对 E 个 expert 的打分，softmax 后取 top-k；
- **experts**：E 个结构相同、参数独立的 FFN；
- **输出**：每个 token 的 k 个 expert 输出，按路由权重加权求和。

Qwen3-MoE 的配置里还有两个影响结构的字段：

- `decoder_sparse_step`：每隔几层用一次 MoE，其余层用普通 MLP；
- `mlp_only_layers`：强制使用普通 MLP 的层号列表。

### 2. 对推理引擎的影响

| 特点 | 对引擎的影响 |
|---|---|
| **总参数多、激活参数少**（如 30B 总参数，每 token 只激活约 3B） | 所有 expert 的权重都必须常驻显存，**留给 KV cache 的显存变少**；但每个 token 的计算量接近一个小模型 |
| **路由是数据相关的** | 每个 expert 本步处理多少 token 事先不知道，**形状是动态的**，和 CUDA Graph 冲突 |
| **一个 batch 的 token 分散到很多 expert** | 如果按 expert 循环，每个 expert 只算几个 token，GEMM 极小、kernel 启动次数多。**需要 fused/grouped GEMM** 把所有 expert 放进一次 kernel |
| **decode 时 batch 很小** | 一步可能要读取很多 expert 的权重，访存压力比 dense 模型更大 |
| **expert 数量多** | 多卡时除了张量并行（TP），还可以把不同 expert 放到不同卡上，即**专家并行（EP）**，这需要 all-to-all 通信 |
| **负载不均衡** | 热门 expert 处理的 token 多，冷门 expert 处理的少，影响 kernel 效率和 EP 时的卡间均衡 |

### 3. 好消息：调度层几乎不用动

回顾 [basic/overview.md](../basic/overview.md) 中"模型代码不感知 batch 结构"这条原则：Scheduler、BlockManager、Sequence 只关心 token 数和 KV 块，**不关心 FFN 是 dense 还是 MoE**。MoE 只改变每层 FFN 内部的计算方式，KV cache 的结构（只取决于 attention 部分）也不变。

所以支持 MoE，改动集中在**模型层、算子层、权重加载，以及执行层中与模型选择、CUDA Graph、显存规划相关的部分**。这也是当前分层设计的价值所在。

---

## 二、需要改动的位置总览

```
┌──────────────────────────────────────────────────────────────────┐
│ 接口层 / 引擎层                                       不变        │
├────────────────────────────┬─────────────────────────────────────┤
│ 调度层（Scheduler 等）  不变 │ 执行层（ModelRunner）               │
│                            │  ① 按 architectures 选择模型类       │
│                            │  ② warmup 要覆盖 MoE 的中间 buffer   │
│                            │  ③ CUDA Graph 兼容（MoE 不能有 host 同步）│
├────────────────────────────┴─────────────────────────────────────┤
│ 模型层   新增 models/qwen3_moe.py                                  │
│          ④ DecoderLayer 按 decoder_sparse_step / mlp_only_layers  │
│             选择 MLP 或 SparseMoeBlock                             │
├──────────────────────────────────────────────────────────────────┤
│ 算子层   新增 layers/fused_moe/                                    │
│          ⑤ router：gate 线性层 + top-k softmax                      │
│          ⑥ experts：按 expert 重排 token + fused grouped GEMM        │
│          ⑦ 并行：TP 下的 all_reduce，或 EP 下的 dispatch/combine      │
├──────────────────────────────────────────────────────────────────┤
│ 横切     ⑧ 模型注册表 registry；loader 支持 expert 权重映射          │
└──────────────────────────────────────────────────────────────────┘
```

下面先看 MS2 怎么做，再逐项讨论设计。

---

## 三、MS2 的实现

### 1. 调用链

```
Qwen3MoeDecoderLayer
 └─ Qwen3MoeSparseMoeBlock                         models/qwen3_moe.py
     ├─ gate: ReplicatedLinear(hidden, E)          → router_logits [N, E]
     └─ experts: SharedFusedMoE → FusedMoE          layers/fused_moe/layer.py
         ├─ router.select_experts                   router/fused_topk_router.py
         │    Triton kernel：softmax + 逐个取 top-k（可选重归一化）
         │    → topk_weights [N, k], topk_ids [N, k]
         └─ quant_method.apply                      unquantized_fused_moe_method.py
             └─ FusedMoEModularKernel               modular_kernel.py
                 ├─ prepare_finalize = MoEPrepareAndFinalizeNoEP   （无 EP、无量化，基本是空操作）
                 └─ fused_experts   = TritonExperts               fused_moe.py
                     ├─ moe_align_block_size：按 expert 重排 token
                     ├─ fused_moe_kernel #1：x @ W13（gate+up）  → [N·k, 2I]
                     ├─ SiluAndMul                               → [N·k, I]
                     ├─ fused_moe_kernel #2：@ W2，乘以路由权重     → [N·k, H]
                     └─ view(N, k, H).sum(dim=1)                  → [N, H]
```

这条链路的类名和分层，基本照搬了 vLLM 的结构（MS2 的 [docs/Moe.md](https://github.com/banfeb/nano-vLLM-MS2/blob/main/docs/Moe.md) 中给出了 vLLM 0.15 的对应调用链）。

### 2. 核心技巧：按 expert 重排 token，再做一次 grouped GEMM

最朴素的做法是对每个 expert 循环：找出选中它的 token，单独做一次 FFN。MS2 中保留了这个版本（`Qwen3MoeSparseMoeBlockOld`）作为对照。问题是 E 次小 GEMM、E 次 kernel 启动，GPU 利用率很低。

fused MoE 的做法是：

```
topk_ids（N=4 个 token，每个选 k=3 个 expert，block_size=4，E=4）
  [[2,3,4],[1,2,4],[1,3,4],[1,2,3]]   （示例来自 MS2 源码注释，expert 编号从 1 开始）

① 展平：每个 (token, expert) 对是一个"槽位"，共 N·k = 12 个
② 按 expert 编号把槽位排序，同一个 expert 的槽位放在一起
③ 每个 expert 的槽位数补齐到 block_size 的整数倍（用一个无效编号填充）
④ 得到 sorted_token_ids，以及每个 block 属于哪个 expert（expert_ids）

kernel 中：每个 program 处理一个 block（block_size 个槽位，它们属于同一个 expert）
           → 读取这个 expert 的权重，做一个 [block_size, K] @ [K, N] 的 GEMM
```

这样一次 kernel 启动就完成了所有 expert 的计算，每个 tile 内部仍然是规整的稠密 GEMM。第二个 GEMM 在写回时顺便乘上路由权重（`mul_routed_weight=True`），最后对 k 个 expert 的结果求和。

### 3. 抽象层次（沿用 vLLM 的设计）

| 抽象 | 职责 | 扩展点 |
|---|---|---|
| `FusedMoE` | 持有 expert 权重、router，对外暴露 `forward(hidden, router_logits)` | 不同模型共用 |
| `quant_method`（`UnquantizedFusedMoEMethod`） | 决定权重格式和使用哪个 kernel | 加入 FP8、INT4 等量化方式 |
| `FusedMoEModularKernel` | 把 MoE 拆成三步：prepare → experts → finalize | — |
| `FusedMoEPrepareAndFinalize` | prepare：激活值量化，或 EP 时把 token 发往对应的卡（dispatch）；finalize：合并结果，或 EP 时收回 token（combine） | 加入 EP |
| `FusedMoEPermuteExpertsUnpermute`（`TritonExperts`） | 真正的计算：重排、GEMM、激活、GEMM、合并 | 换成 CUTLASS 等后端 |

这套抽象的价值在于：**量化、并行方式、计算后端三个维度可以独立替换**。对 nano-vllm 这种教学型项目来说有些重，但它为以后加入 EP 和量化留出了位置。

### 4. 权重的组织与加载

MS2 让每个 expert 成为一个独立的子模块，直接复用当前项目已有的并行线性层：

```python
class _ExpertMLP(nn.Module):
    gate_up_proj = MergedColumnParallelLinear(hidden, [I, I])
    down_proj    = RowParallelLinear(I, hidden)

class FusedMoE(nn.Module):
    for e in range(E):
        self.add_module(str(e), _ExpertMLP(...))     # 参数名形如 experts.0.gate_up_proj.weight
```

好处是**加载逻辑不用改**。checkpoint 中的 `mlp.experts.3.gate_proj.weight` 经过现有的 `packed_modules_mapping`（`gate_proj → gate_up_proj` 的第 0 片），就能找到 `mlp.experts.3.gate_up_proj.weight`，并由 `MergedColumnParallelLinear.weight_loader` 完成 TP 切分。

kernel 需要的是把所有 expert 堆叠起来的 3 维张量，MS2 用两个 property 临时拼出来：

```python
@property
def w13_weight(self):   # [E, H, 2I]
    return torch.stack([e.gate_up_proj.weight.transpose(0, 1) for e in experts]).contiguous()
```

### 5. 模型注册表

MS2 新增了 [registry.py](https://github.com/banfeb/nano-vLLM-MS2/blob/main/nanovllm/models/registry.py)，并在 loader 中按 `hf_config.architectures[0]` 动态 import 对应的模型类：

```python
NANO_VLLM_MODELS = {
    "Qwen3ForCausalLM":    ("qwen3",     "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),
}
```

这正好解决了 [basic/overview.md](../basic/overview.md#新增一个模型需要做什么) 中提到的"模型类写死在 ModelRunner 里"的问题。

---

## 四、MS2 实现中值得改进的地方

下面几点都是阅读 MS2 源码得出的，其中第 2 点有 MS2 自己的注释佐证，其余几点是推断。它们也正是在当前项目中落地时需要重点设计的地方。

### 1. 每次前向都要重新堆叠所有 expert 的权重

`w13_weight` / `w2_weight` 是 property，**每一层、每一步前向都会执行一次 `stack + transpose + contiguous`**：

- 每步都要把所有 expert 的权重完整拷贝一遍。对 MoE 来说，这是全部参数中最大的那部分；
- 拷贝期间会额外占用一份与 expert 权重同样大小的临时显存。warmup 测出的峰值会把它算进去，导致留给 KV cache 的显存变少。

**改进**：在初始化时就按 kernel 需要的布局分配好堆叠后的参数，加载时直接写入对应的切片：

```python
self.w13_weight = nn.Parameter(torch.empty(E_local, 2 * I_local, H))   # 按 kernel 需要的布局
self.w2_weight  = nn.Parameter(torch.empty(E_local, H, I_local))
self.w13_weight.weight_loader = self.weight_loader                     # 签名多一个 expert_id

# loader 的映射需要多一个维度：
#   "experts.{e}.gate_proj" → ("experts.w13_weight", expert_id=e, shard=0)
#   "experts.{e}.up_proj"   → ("experts.w13_weight", expert_id=e, shard=1)
#   "experts.{e}.down_proj" → ("experts.w2_weight",  expert_id=e)
```

vLLM 的 `FusedMoE.make_expert_params_mapping` 就是这样做的。代价是 [loader.py](../../nanovllm/utils/loader.py) 要支持"一个 checkpoint 权重名映射到参数的某个 expert 切片"，需要扩展 `packed_modules_mapping` 的语义。

### 2. 重排 token 的过程在 Python 中逐个 expert 执行，与 CUDA Graph 不兼容

MS2 的 `moe_align_block_size` 对每个 expert 执行一次 `nonzero().tolist()`：

- 每个 expert 一次 GPU→CPU 同步，E = 128 时每层就是 128 次；
- 存在 host 同步，就无法被 CUDA Graph 录制。MS2 的 [bench.py](https://github.com/banfeb/nano-vLLM-MS2/blob/main/bench.py) 注释中也明确写了"moe 暂时未能支持图捕获优化"，所以 MoE 的 benchmark 都使用 `enforce_eager=True`。

对 decode 来说这影响很大：[basic/ModelRunner.md](../basic/ModelRunner.md) 中提到，decode 阶段的瓶颈本来就是 kernel 启动开销，CUDA Graph 是最重要的优化之一。

**改进**：

- 用 GPU kernel 实现重排（vLLM 中有专门的 `moe_align_block_size` CUDA 算子，也可以用 Triton 写一个"计数 → 前缀和 → 填充"的版本），整个过程不回到 CPU；
- **输出 buffer 按最坏情况分配固定大小**（`N·k + E·(block_size-1)`，再向上取整），真实长度用一个 GPU 张量 `num_tokens_post_pad` 表示，kernel 根据它提前退出。这样形状就只取决于 batch size，可以和现有的"按 batch size 分档"的 CUDA Graph 机制配合。

### 3. 张量并行下可能缺少 all_reduce

MS2 中每个 expert 的 `gate_up_proj` 按输出维度切分、`down_proj` 按输入维度切分，和 dense MLP 一样。但 dense MLP 中的 all_reduce 是在 `RowParallelLinear.forward` 里完成的，而 fused kernel 直接使用堆叠后的权重，**绕过了 `RowParallelLinear.forward`**。在 MS2 的 MoE 路径中，我没有找到 all_reduce。

因此按代码推断，TP > 1 时，每张卡得到的只是部分和，结果不正确。MS2 的示例和 benchmark 都是 `tensor_parallel_size=1`，所以没有暴露出来。

**改进**：在 `FusedMoE.forward` 末尾，`tp_size > 1` 时显式调用 `dist.all_reduce`。或者更进一步，把这一步放进 `PrepareAndFinalize.finalize`，让"TP 合并"和"EP 合并"走同一个扩展点。

### 4. 路由的配置项没有从模型 config 读取

MS2 的 router 固定使用 `renormalize=True`（top-k 权重重新归一化）。Qwen3-MoE 的 config 中有 `norm_topk_prob` 字段，不同 MoE 模型的设置可能不同；路由函数也可能是 sigmoid 而不是 softmax（DeepSeek-V3）。

**改进**：把 `renormalize`、`scoring_func` 作为 `FusedMoE` 的构造参数，由模型层从 hf_config 中读取后传入。

### 5. 共享 expert 只有占位

MS2 有 `SharedFusedMoE` 类，但目前只是直接调用父类。Qwen3-MoE 没有共享 expert，但 Qwen2-MoE、DeepSeek 系列有：每个 token 除了 top-k 个路由 expert，还要经过一个所有 token 共享的 expert。

**改进**：`SharedFusedMoE` 持有一个普通 MLP 作为共享 expert，`forward` 中将两部分结果相加。共享 expert 的计算和路由 expert 的计算互不依赖，进一步还可以放在不同的 CUDA stream 上并行执行。

---

## 五、在当前项目中的落地设计

### 1. 文件级改动清单

| 层 | 文件 | 改动 |
|---|---|---|
| 横切 | 新增 `nanovllm/models/registry.py` | `architectures` → (模块名, 类名) |
| 执行层 | [model_runner.py](../../nanovllm/engine/model_runner.py) | 不再写死 `Qwen3ForCausalLM`，改为从注册表查找 |
| 模型层 | 新增 `nanovllm/models/qwen3_moe.py` | Attention 部分和 qwen3 相同（可直接复用 `Qwen3Attention`）；DecoderLayer 按 `decoder_sparse_step` / `mlp_only_layers` 选择 `Qwen3MLP` 或 `Qwen3MoeSparseMoeBlock` |
| 算子层 | 新增 `nanovllm/layers/fused_moe/` | `FusedMoE`（堆叠后的参数 + weight_loader）、router、`moe_align_block_size`（GPU 版）、fused GEMM Triton kernel |
| 横切 | [loader.py](../../nanovllm/utils/loader.py) | 支持 expert 维度的权重映射 |
| 执行层 | `warmup_model` | 无需修改，但要确认 warmup 走过了 MoE 路径（它会正常执行一次完整的前向） |
| 调度层 | — | 不需要改动 |

### 2. 并行策略的选择

| 方案 | 做法 | 通信 | 适用场景 |
|---|---|---|---|
| **TP（每个 expert 内部切分）** | 每张卡持有所有 expert 的 1/tp 中间维度 | 每层一次 all_reduce，和 dense 模型相同 | expert 数量不多、卡数少（单机 2～8 卡） |
| **EP（按 expert 分配到不同卡）** | 每张卡持有 E/ep 个完整的 expert | dispatch 和 combine 两次 all-to-all | expert 数量多、卡数多；单个 expert 的中间维度很小，再做 TP 切分会让 GEMM 太窄 |
| TP + EP 混合 | attention 用 TP，MoE 用 EP | 两种都有 | 大规模部署 |

对 nano-vllm 来说，**建议先只做 TP**：它和现有的并行模型完全一致，只需要在 MoE 输出后加一次 all_reduce。同时保留 `PrepareAndFinalize` 这个扩展点，将来加入 EP 时，只需新增一个实现 dispatch/combine 的子类，模型层不用改。

注意一个约束：TP 切分的是每个 expert 的中间维度 `moe_intermediate_size`。它通常很小（Qwen3-MoE 中远小于 dense 模型的 `intermediate_size`），切分之后每张卡上的 GEMM 会非常窄。这正是大规模部署时更倾向于用 EP 的原因。

### 3. 与 CUDA Graph 的配合

MoE 层要能被 CUDA Graph 录制，必须满足：

1. **整个前向过程中没有 host 同步**：不能有 `.tolist()`、`.item()`、依赖 GPU 数据的 Python 分支（例如 `if mask.sum() == 0`）；
2. **所有中间 buffer 的形状只取决于 token 数**：`sorted_token_ids`、`expert_ids` 等按最坏情况分配；
3. **kernel 的 grid 大小固定**：按最坏情况的 block 数启动，多出来的 program 读取 `num_tokens_post_pad` 后直接返回。

满足这三点后，就可以沿用现有的 `capture_cudagraph`，不需要任何改动。

### 4. 显存规划

- **权重**：所有 expert 常驻显存，`allocate_kv_cache` 中的 `used` 会自动把它算进去，不需要额外处理；
- **激活峰值**：MoE 的中间张量是 `[N·k, 2I]` 和 `[N·k, H]`，比 dense MLP 的 `[N, 2·intermediate]` 可能更大，也可能更小，取决于 k 和 I。warmup 用 `max_num_batched_tokens` 跑一遍完整的前向，峰值会被自动测出来；
- **只要采纳第四节第 1 点的改进**（不在前向中堆叠权重），就不会有额外的大块临时显存。

### 5. 推荐的实现顺序

1. 注册表 + `qwen3_moe.py`，先用"按 expert 循环"的朴素版本跑通，验证结果正确（和 HuggingFace 对比输出）；
2. 引入堆叠后的参数和新的 weight_loader，替换为 fused Triton kernel；
3. 把 `moe_align_block_size` 改成 GPU 版本，去掉 host 同步，打开 CUDA Graph；
4. 加入 TP 的 all_reduce，并在多卡上验证；
5. （可选）量化、EP、共享 expert。

每一步都能单独验证正确性和性能，出问题时容易定位。
