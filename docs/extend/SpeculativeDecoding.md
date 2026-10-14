# 扩展二：让 nano-vllm 支持推测解码（Speculative Decoding）

本文是在当前 nano-vllm 上实现推测解码的设计文档。目标是用**一套统一的框架**同时支持两种草稿来源：

- **n-gram（prompt lookup）**：从上下文中查找重复片段作为草稿，零额外显存；
- **草稿小模型（draft model）**：用同系列的小模型（如 Qwen3-0.6B）为大模型（如 Qwen3-8B）打草稿。

两种方法共用同一套调度、验证、拒绝采样和后处理，只在"草稿从哪里来"这一步不同，通过配置切换。

内容分为：

1. 原理：为什么能加速，怎样保证输出分布不变；
2. 设计目标，以及推测解码对引擎各层提出的新要求；
3. 配置设计；
4. 整体架构；
5. 各组件的详细设计；
6. 设计取舍：考虑过但没有采用的方案；
7. 与现有机制的配合、文件改动清单、实现顺序。

阅读前建议先看 [basic/Scheduler.md](../basic/Scheduler.md)、[basic/block.md](../basic/block.md) 和 [basic/Sequence.md](../basic/Sequence.md)。推测解码几乎会碰到这几个组件的每一个设计点。

---

## 一、原理

### 1. 为什么能加速

[basic/ContinuousBatching.md](../basic/ContinuousBatching.md) 中讲过：decode 阶段每步只生成 1 个 token，却要把整套权重从显存读一遍，瓶颈在**访存**而不在计算。

所以，**一次前向算 1 个 token 和算 K+1 个 token，耗时几乎一样**。推测解码利用了这一点：

```
普通 decode：  每步输入 1 个 token → 产出 1 个 token

推测解码：     ① propose：用一个便宜的方法"猜"出后面 K 个 token（草稿）
              ② verify： 把"最后一个 token + K 个草稿"一起送进大模型，一次前向得到 K+1 个位置的分布
              ③ accept： 从前往后逐个检查草稿，接受连续的一段，在第一个被拒绝的位置补一个 token
              → 每步产出 1 ~ K+1 个 token，而验证的耗时和普通 decode 差不多
```

### 2. 两种草稿来源

| | n-gram | 草稿小模型 |
|---|---|---|
| 做法 | 在已有上下文中找和"最后 n 个 token"相同的片段，把它后面的 token 当作草稿 | 小模型自回归采样 K 个 token |
| 草稿分布 q | one-hot（草稿是确定的） | 完整的词表分布 |
| 额外显存 | 无 | 小模型权重 + 它自己的 KV cache |
| 每步额外开销 | CPU 上的字符串匹配 | K 次小模型前向 |
| 前提 | 无 | 两个模型**词表一致**（同一个 tokenizer） |
| 适合的场景 | 输出大量复制输入：改代码、摘要、基于检索内容的问答 | 通用，开放式生成也有较高的接受率 |

草稿小模型是推测解码最原始的形式（Leviathan et al. 2023；Chen et al. 2023）。n-gram 可以看作它的特例：草稿分布 q 退化为 one-hot。正因为如此，两者可以共用同一套验证逻辑。

### 3. 怎样保证输出分布不变：拒绝采样

设第 i 个位置大模型的分布为 p，草稿分布为 q，草稿 token 为 d：

```
以概率 min(1, p(d) / q(d)) 接受 d
若拒绝：从修正分布 r(x) ∝ max(0, p(x) − q(x)) 中采样一个 token 作为替代，本轮到此结束
若 K 个草稿全部被接受：再从最后一个位置的 p 中额外采样一个 token（bonus token）
```

可以证明，这样得到的每个 token 都服从 p，也就是和只用大模型逐个采样**完全同分布**。

n-gram 的 q 是 one-hot（`q(d) = 1`），公式自然退化为：

- 接受概率就是 **p(d)**；
- 拒绝后从 **p 去掉 d 之后重新归一化**的分布中采样。

所以统一的拒绝采样器只需要做一件事：**没有 q 时，把它当作 one-hot**（见第五节第 3 小节）。

### 4. 收益估算

设每个草稿被接受的概率都是 α（简化为互相独立），每步提议 K 个草稿，则每步期望产出的 token 数为：

```
E[tokens/step] = 1 + α + α² + … + α^K = (1 − α^(K+1)) / (1 − α)
```

每一步也更贵了。设生成一个草稿 token 的耗时是大模型一次 decode 的 c 倍，batch 较小时验证的耗时约等于一次普通 decode，则：

```
一步的耗时 ≈ (K·c + 1) × T_decode
加速比     ≈ E[tokens/step] / (K·c + 1)
```

例如 α = 0.7、K = 5 时，E ≈ 2.94：

- **n-gram**：c ≈ 0，理论加速约 2.9×，前提是真能找到匹配（α 才会这么高）；
- **0.6B 草稿 + 8B 目标**：按参数量算 c 约 0.075，但小模型的耗时更多由 kernel 启动等固定开销决定，c 大约在 0.1 ~ 0.2，加速约 1.5 ~ 2.0×。

batch 增大后，有两项开销会变得显著：

- **验证**：decode 逐渐从访存瓶颈变成计算瓶颈，一次验证 B·(K+1) 个 token 的耗时明显上升，被拒绝的草稿白白占用了算力；
- **草稿模型的 K 次前向是串行的**，batch 越大越贵。

所以推测解码在**小 batch、对延迟敏感**的场景中收益最大，大 batch 时甚至会变慢。设计中要支持**按 batch 大小自动关闭**（见第五节第 8 小节），阈值通过实测确定。

---

## 二、设计目标与新要求

### 1. 设计目标

| 目标 | 说明 |
|---|---|
| **一套框架，两种方法** | n-gram 和草稿模型共用调度、验证、拒绝采样和后处理，只有 propose 不同 |
| **配置切换** | 通过一个 `speculative_config` 选择方法和参数；不配置时行为与现在完全相同 |
| **无损** | 输出分布与不开推测解码时完全一致 |
| **保持 CUDA Graph** | 验证步和草稿模型的前向都要能录图，否则省下的步数会被 kernel 启动开销吃掉 |
| **改动最小** | 尽量复用现有的进度字段、`hash_blocks`、decode 的 attention 路径，保持 nano-vllm 简洁的风格 |
| **保持现有分工** | Scheduler 是唯一修改序列状态的地方；ModelRunner 只读输入、返回结果 |

### 2. 推测解码对引擎各层的要求

推测解码**几乎触及每一层**。原因是它打破了 decode 阶段一个隐含的前提："每条序列每步只算 1 个 token，也只产出 1 个 token"。

| 层 | 原来的前提 | 推测解码之后 |
|---|---|---|
| **Config** | 只有一个模型 | 新增推测解码的配置；草稿模型方法需要第二个模型的 hf_config |
| **Sequence** | decode 时 `num_scheduled_tokens = 1` | 每步计算 1 + K 个 token，但只确认其中一部分 |
| **BlockManager** | 每步最多新增 1 个块 | 一次为 1 + K 个 token 预留空间 |
| **prefix cache** | 写满的块一定是有效内容 | 被拒绝的 token 的 KV 也写进了 cache，**绝不能登记 hash** |
| **Scheduler** | 每步 token 数 = 序列数 | 每步 token 数 = B·(1 + K)；后处理一次追加多个 token，EOS 可能出现在中间 |
| **ModelRunner** | decode 的 Q 长度为 1 | 验证时 Q 长度为 1 + K；草稿模型方法还要管理第二个模型及其 KV cache |
| **Attention** | decode 时 q 为 `(B, 1, H, D)` | q 为 `(B, q_len, H, D)` |
| **Sampler** | 每条序列采样一个 token | 草稿模型要返回概率 q；验证时做拒绝采样，每条序列输出 1 ~ K+1 个 token |
| **TP 通信** | 每步传输 `seqs` | 草稿 token 也要让所有 rank 知道 |
| **新组件** | — | NgramProposer、RejectionSampler |

---

## 三、配置设计

### 1. 使用方式

沿用 vLLM 的接口风格，用一个 `speculative_config` 参数配置，可以传 dict：

```python
from nanovllm import LLM, SamplingParams

# 不开启：与现在完全相同
llm = LLM("~/huggingface/Qwen3-8B")

# n-gram
llm = LLM("~/huggingface/Qwen3-8B",
          speculative_config=dict(method="ngram", num_speculative_tokens=4,
                                  prompt_lookup_min=2, prompt_lookup_max=5))

# 草稿小模型
llm = LLM("~/huggingface/Qwen3-8B",
          speculative_config=dict(method="draft_model", model="~/huggingface/Qwen3-0.6B",
                                  num_speculative_tokens=5))
```

`LLMEngine.__init__` 会按 `Config` 的字段名过滤 `**kwargs`，所以只要 `Config` 中有 `speculative_config` 这个字段，上面的写法就能直接透传，引擎不需要改。

### 2. 配置类

[config.py](../../nanovllm/config.py) 中新增 `SpeculativeConfig`，并在 `Config` 中增加一个字段：

```python
@dataclass(slots=True)
class SpeculativeConfig:
    method: str = "ngram"                    # "ngram" | "draft_model"
    num_speculative_tokens: int = 4          # 每步提议的草稿数 K
    model: str | None = None                 # draft_model：草稿模型路径
    prompt_lookup_min: int = 2               # ngram：最短匹配长度
    prompt_lookup_max: int = 5               # ngram：最长匹配长度
    max_batch_size: int = 32                 # batch 超过它时本步不推测，走普通 decode
    hf_config: AutoConfig | None = None      # draft_model：草稿模型的 config，自动读取

    def __post_init__(self):
        assert self.method in ("ngram", "draft_model")
        assert self.num_speculative_tokens > 0
        if self.method == "draft_model":
            self.model = os.path.expanduser(self.model)
            assert os.path.isdir(self.model)
            self.hf_config = AutoConfig.from_pretrained(self.model)
        else:
            assert 1 <= self.prompt_lookup_min <= self.prompt_lookup_max


@dataclass(slots=True)
class Config:
    ...
    speculative_config: SpeculativeConfig | dict | None = None

    def __post_init__(self):
        ...
        if isinstance(self.speculative_config, dict):
            self.speculative_config = SpeculativeConfig(**self.speculative_config)
        if (spec := self.speculative_config) is not None:
            k = spec.num_speculative_tokens
            assert self.max_num_seqs * (1 + k) <= self.max_num_batched_tokens   # 验证步的激活不超过 warmup 测出的峰值
            assert self.tensor_parallel_size == 1                               # 第一版只支持单卡，见第五节第 10 小节
            if spec.method == "draft_model":
                assert spec.hf_config.vocab_size == self.hf_config.vocab_size   # 拒绝采样按 token id 对齐 p 和 q
```

### 3. 字段说明

| 字段 | 适用方法 | 说明 |
|---|---|---|
| `method` | 共用 | 选择草稿来源 |
| `num_speculative_tokens` | 共用 | K。草稿模型的经验值为 3 ~ 5；n-gram 的草稿几乎免费，可以稍大，但验证的开销随 K 线性增长 |
| `max_batch_size` | 共用 | 当前 running 的序列数超过它时，本步不推测。默认值需要实测确定（第七节第 3 小节第 6 步） |
| `model` | draft_model | 草稿模型路径，必须与主模型使用同一个 tokenizer |
| `prompt_lookup_min` / `prompt_lookup_max` | ngram | 匹配长度的范围，从长到短尝试 |
| `hf_config` | draft_model | 内部字段，自动从 `model` 读取 |

### 4. 命令行

[example.py](../../example.py) 和 [bench.py](../../bench.py) 增加对应的参数，拼成 dict 传给 `LLM`：

```
--spec-method {ngram,draft_model}  --num-spec-tokens 4
--spec-model ~/huggingface/Qwen3-0.6B
--prompt-lookup-min 2  --prompt-lookup-max 5
```

---

## 四、整体架构

### 1. 组件关系

```
                        LLMEngine.step()
                              │
            ┌─────────────────┼──────────────────────────┐
            ▼                 ▼                          ▼
     Scheduler.schedule   ModelRunner.run / run_spec   Scheduler.postprocess / postprocess_spec
     ├ 决定本步是否推测     │
     └ 预留 1+K 个位置      ├─ ① propose ──┬─ method = ngram       → NgramProposer（CPU，rank 0）
                           │              └─ method = draft_model → 草稿模型 K 次前向（GPU）
                           │                                        ↑ 两种方法唯一不同的地方
                           ├─ ② verify   ：大模型一次前向，Q = K+1
                           └─ ③ accept   ：RejectionSampler（q 为 None 时按 one-hot 处理）
                                           返回每条序列的 token 列表，不修改 seq
```

### 2. 一个推测解码 step 的完整流程

```
LLMEngine.step()
 │
 ├─ Scheduler.schedule()
 │    prefill 分支：不变（草稿模型方法在 ModelRunner 中让两个模型一起 prefill）
 │    decode 分支：本步是否推测？ → n = 1 + K 或 1
 │                 每条序列：can_append(seq, n) → may_append(seq, n)，不够时抢占
 │                 seq.num_scheduled_tokens = n
 │
 ├─ 推测步：ModelRunner.call("run_spec", seqs)
 │    ① propose：得到 draft_ids [B, K]、num_drafts [B]、draft_probs [B, K, V] 或 None
 │    ② verify： 输入 [last_token, d1..dK]，一次前向得到 B·(K+1) 行 logits
 │    ③ accept： rank 0 上做拒绝采样
 │    返回 list[list[int]]：每条序列 = 被接受的草稿 + 1 个修正 / bonus token
 │
 └─ Scheduler.postprocess_spec(seqs, token_ids)
      逐个追加 → 修正 num_scheduled_tokens → hash_blocks → 推进 num_cached_tokens → 追加修正 token
```

`LLMEngine.step()` 通过 `num_scheduled_tokens` 区分普通 decode 和推测步，不需要新的返回值：

```python
seqs, is_prefill = self.scheduler.schedule()
if is_prefill or seqs[0].num_scheduled_tokens == 1:
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
else:
    token_ids = self.model_runner.call("run_spec", seqs)
    self.scheduler.postprocess_spec(seqs, token_ids)
```

---

## 五、各组件的详细设计

### 1. 统一的 propose 输出

无论哪种方法，propose 都输出同样形状的三样东西：

| 输出 | 形状 | n-gram | 草稿模型 |
|---|---|---|---|
| `draft_ids` | `[B, K]` int64 | 匹配到的 token，不足 K 个的部分随便填（例如 0） | 采样出的 K 个 token |
| `num_drafts` | `[B]` int64 | 每条序列实际的草稿数，0 ~ K | 全部为 K |
| `draft_probs` | `[B, K, V]` 或 `None` | `None`，表示 one-hot | 每一步采样时的完整分布 q |

验证步的形状**永远是 `[B, K+1]`**，草稿不足 K 个的序列用补齐处理（第五节第 9 小节）。这样 verify 的 CUDA Graph 只需要按 batch size 分档，不用关心每条序列有多少草稿。

### 2. n-gram：NgramProposer

新增 `nanovllm/spec_decode/ngram_proposer.py`。只在 rank 0 上创建和调用；rank 0 的 ModelRunner 和 LLMEngine 在同一个进程中，拿到的是完整的 Sequence 对象，可以直接读 `token_ids`。

```python
class NgramProposer:

    def __init__(self, k: int, n_min: int, n_max: int):
        self.k, self.n_min, self.n_max = k, n_min, n_max

    def propose(self, seqs: list[Sequence]):
        draft_ids = torch.zeros(len(seqs), self.k, dtype=torch.int64)
        num_drafts = torch.zeros(len(seqs), dtype=torch.int64)
        for i, seq in enumerate(seqs):
            draft = self.lookup(np.asarray(seq.token_ids))
            draft_ids[i, :len(draft)] = torch.from_numpy(draft)
            num_drafts[i] = len(draft)
        return draft_ids.cuda(non_blocking=True), num_drafts.cuda(non_blocking=True), None

    def lookup(self, tokens: np.ndarray) -> np.ndarray:
        for n in range(self.n_max, self.n_min - 1, -1):          # 从长到短，长匹配更可靠
            if len(tokens) <= n:
                continue
            windows = sliding_window_view(tokens[:-1], n)         # 不包括末尾自身
            hits = np.flatnonzero((windows == tokens[-n:]).all(axis=1))
            if len(hits):
                start = hits[-1] + n                              # 取最近的一次出现
                return tokens[start:start + self.k]
        return tokens[:0]
```

这是最直接的 O(L·n) 写法，符合 nano-vllm 的风格。如果长序列的匹配成为瓶颈，可以换成 KMP 或用 Numba 加速，接口不变。

### 3. 统一的拒绝采样：RejectionSampler

新增 `nanovllm/spec_decode/rejection_sampler.py`，只在 rank 0 上运行（logits 只 gather 到 rank 0）。两种方法的差异被收敛到两条规则：

- **`draft_probs is None`**：q 按 one-hot 处理，`q(d) = 1`；修正分布为 p 去掉 d；
- **位置 j ≥ `num_drafts`**（n-gram 草稿不足）：这个位置没有草稿，强制"拒绝"，并且直接从 p 采样，和 bonus token 的处理相同。

```python
class RejectionSampler(nn.Module):

    def forward(self, logits, temperatures, draft_ids, num_drafts, draft_probs=None):
        B, K = draft_ids.shape
        p = torch.softmax(logits.float() / temperatures.repeat_interleave(K + 1)[:, None], -1).view(B, K + 1, -1)
        p_d = p[:, :K].gather(2, draft_ids[..., None]).squeeze(-1)                   # p(d)
        q_d = 1. if draft_probs is None else draft_probs.gather(2, draft_ids[..., None]).squeeze(-1)
        valid = torch.arange(K, device=p.device) < num_drafts[:, None]               # 有草稿的位置
        accepted = (torch.rand_like(p_d) * q_d < p_d) & valid                         # rand < p/q
        num_accepted = accepted.int().cumprod(1).sum(1)                              # 第一个拒绝之后全部作废

        # 下一个 token 的分布：第一个拒绝的位置，或者 bonus 位置 K
        rows = torch.arange(B, device=p.device)
        p_next = p[rows, num_accepted]
        rejected = num_accepted < num_drafts                                         # 确实拒绝了一个草稿
        j = num_accepted.clamp(max=K - 1)
        if draft_probs is None:
            q_next = F.one_hot(draft_ids[rows, j], p.size(-1)).float()
        else:
            q_next = draft_probs[rows, j]
        residual = (p_next - q_next).clamp_min(0)
        next_probs = torch.where(rejected[:, None], residual, p_next)               # 没有草稿 / 全部接受：直接用 p
        next_ids = torch.multinomial(next_probs / next_probs.sum(-1, keepdim=True), 1).squeeze(1)

        draft_ids, num_accepted, next_ids = draft_ids.tolist(), num_accepted.tolist(), next_ids.tolist()
        return [draft_ids[i][:num_accepted[i]] + [next_ids[i]] for i in range(B)]
```

要点：

- 用 `rand * q < p` 代替 `rand < p / q`，避免除以 0；
- **温度要一致**：草稿模型采样、计算 q 时使用与大模型相同的 temperature，否则 p/q 的比值没有意义；
- 当前项目的 `SamplingParams` 不允许 temperature = 0。如果将来支持 greedy，验证就退化为"argmax 是否等于草稿"，可以跳过随机数。

### 4. 草稿模型的 KV cache：与大模型共用一张 block_table

```
                       seq.block_table = [7, 2, 9]（只有一张）
                                  │
             ┌────────────────────┴────────────────────┐
     大模型 kv_cache[:, :, 7]                  草稿模型 draft_kv_cache[:, :, 7]
     （按大模型的层数、头数）                    （按草稿模型的层数、头数）
```

两个模型在每次前向时使用**同一份** `slot_mapping` 和 `block_tables`，各自把 KV 写进自己的池中相同块号的位置。所以：

- **BlockManager 完全不知道草稿模型的存在**，分配、释放、抢占只管理一套块；
- **prefix cache 自动对两个模型生效**：命中的块在两个池中都有该前缀的 KV；
- **n-gram 方法**没有第二个池，这一节的内容全部跳过。

显存规划：`allocate_kv_cache` 中每个块的字节数改为两个模型之和，两个池的块数相同：

```python
block_bytes = target_block_bytes + draft_block_bytes
config.num_kvcache_blocks = available // block_bytes
self.kv_cache = torch.empty(2, target_layers, num_blocks, block_size, target_kv_heads, target_head_dim)
self.draft_kv_cache = torch.empty(2, draft_layers, num_blocks, block_size, draft_kv_heads, draft_head_dim)
# 分别绑定到两个模型的 Attention 模块的 k_cache / v_cache
```

这里有一个前提：**登记了 hash 的块，两个池中的内容都必须有效**。所以：

- **prefill 时两个模型都要跑**（包括 chunked prefill 的每一个 chunk）。`run()` 的 prefill 分支在大模型前向之后，用同一个 context 再调用一次 `self.draft_model(input_ids, positions)`，只写 KV，不计算 logits；
- **warmup** 走的就是 prefill 路径，草稿模型自然被算进激活峰值；
- decode 阶段由下一小节的"追赶"保证。

### 5. 草稿模型的进度：Q=2 的第 0 步

一个推测步结束后，两个模型的 KV 进度最多相差 1 个 token：

```
step 开始时：已确认 t0..t9，t9 = last_token，两个模型的 KV 都覆盖 t0..t8
草稿模型：输入 t9, d1, ..., d_{K−1}，写入它们的 KV → 生成 d1..dK（dK 从未作为输入）
大模型：  输入 t9, d1, ..., dK，写入它们的 KV
```

- 有草稿被拒绝时，两边的有效 KV 都覆盖到最后一个被接受的草稿，**进度一致**；
- **K 个全部接受时**，大模型有 dK 的 KV，草稿模型没有，**草稿模型落后 1 个 token**。

处理方法：**草稿模型的第 0 次前向总是输入最后 2 个已确认的 token**：

```
第 0 次：  input = [token(len−2), last_token]，positions = [len−2, len−1]，Q = 2 → 采样 d1
第 t 次：  input = [d_t]，position = len−1+t，Q = 1 → 采样 d_{t+1}       （t = 1..K−1）
```

- 落后 1 个时，正好补上位置 len−2 的 KV；
- 没有落后时，位置 len−2 的 KV 被重新计算一遍，结果相同，覆盖写入无害。

这样不需要额外的字段记录草稿模型的进度，第 0 步的形状永远是规整的 `(B, 2)`，可以录制 CUDA Graph。代价是每步多算 1 个 token，对小模型可以忽略。

草稿模型前向时的 `context_lens`、`slot_mapping` 都按上面的位置计算，block_table 直接用 `seq.block_table`（调度时已经预留到 `len + K`）。草稿 token 只存在于 ModelRunner 的局部张量中，不写入 Sequence。

例外：如果序列恰好在"K 个全部接受"的这一步结束，dK 所在位置的草稿 KV 永远不会补上。若该块已登记 hash，之后命中它的请求会读到这个位置的旧数据。这只会让草稿模型猜得差一点；**验证由大模型完成，输出依然无损**。

### 6. 进度字段：先算、再确认

当前项目已经有一套统一的进度表示（见 [basic/Sequence.md](../basic/Sequence.md)）：

- `num_cached_tokens`：KV 已经写入 cache 的 token 数；
- `num_scheduled_tokens`：本步要计算的 token 数；
- ModelRunner 按 `[num_cached_tokens, num_cached_tokens + num_scheduled_tokens)` 准备输入，`hash_blocks` 按同一个范围登记写满的块。

推测解码只需要区分**"本步计算了多少"**和**"本步确认了多少"**：

```
schedule 时：   num_scheduled_tokens = 1 + K          ← 计算范围包含全部草稿
postprocess 时：num_scheduled_tokens = 1 + 接受数      ← 改成确认的数量，再调用 hash_blocks
```

这样 `hash_blocks` 和 `num_cached_tokens` 的推进逻辑完全不用改，被拒绝的 KV 永远不会登记 hash。

`Scheduler.postprocess_spec`：

```python
def postprocess_spec(self, seqs: list[Sequence], token_ids: list[list[int]]) -> int:
    num_new_tokens = 0
    for seq, ids in zip(seqs, token_ids):
        *accepted, next_id = ids
        n = 0
        for token_id in accepted:                          # ① 逐个追加被接受的草稿
            seq.append_token(token_id)
            n += 1
            if self.is_stop(seq, token_id):
                break
        seq.num_scheduled_tokens = 1 + n                   # ② 本步确认的范围：last_token + n 个草稿
        self.block_manager.hash_blocks(seq)                # ③ 只登记确认范围内写满的块
        seq.num_cached_tokens += seq.num_scheduled_tokens  # ④
        seq.num_scheduled_tokens = 0
        if not self.is_stop(seq, seq.last_token):
            seq.append_token(next_id)                      # ⑤ 修正 / bonus token，KV 下一步再算
            n += 1
        num_new_tokens += n
        if self.is_stop(seq, seq.last_token):
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
    return num_new_tokens
```

`is_stop` 就是现有 `postprocess` 中的停止条件（EOS 或达到 `max_tokens`），抽成一个小函数供两处共用。返回值用于 LLMEngine 统计吞吐。

顺序很重要：`hash_blocks` 用 `seq.block(i)` 读取 token 计算哈希，所以必须**先把被接受的草稿追加进 `token_ids`**，再调用它。

例子：block_size = 4，K = 3，已确认 t0..t9（num_cached_tokens = 9），草稿为 d1、d2、d3，d1 被接受、d2 被拒绝，修正 token 为 r2：

```
计算范围：[9, 13)，t9 d1 d2 d3 的 KV 都写入了 cache
① 追加 d1           → token_ids = t0..t9 d1
② num_scheduled = 2 → 确认范围 [9, 11)
③ hash_blocks：start = 9//4 = 2，end = 11//4 = 2 → 本步没有写满的块
④ num_cached_tokens = 11
⑤ 追加 r2           → last_token = r2，下一步从位置 11 开始计算
位置 11、12 中 d2、d3 的 KV 已经无效：下一步会被覆盖，也永远不会被登记 hash
```

### 7. BlockManager：把 append 推广到 n 个 token

```python
def can_append(self, seq: Sequence, n: int = 1) -> bool:
    # 本步写入位置 [num_cached_tokens, num_cached_tokens + n)，block_table 要覆盖到最后一个位置
    needed = (seq.num_cached_tokens + n + self.block_size - 1) // self.block_size - len(seq.block_table)
    return len(self.free_block_ids) >= max(0, needed)

def may_append(self, seq: Sequence, n: int = 1):
    while len(seq.block_table) * self.block_size < seq.num_cached_tokens + n:
        seq.block_table.append(self._allocate_block())
```

n = 1 时与现有逻辑等价（decode 时 `num_cached_tokens = len(seq) − 1`）。

被拒绝后多出来的块**不释放**：

- 下一步还要预留 K 个位置，这个块大概率马上又要用；
- 它最多只有 1 个，序列结束或被抢占时由 `deallocate` 一并释放；
- 它里面只有无效的 KV，但没有登记 hash，不会被别人命中。

### 8. Scheduler：decode 分支

```python
def schedule(self):
    ...  # prefill 分支不变

    # decode
    spec = self.spec_config
    n = 1
    if spec and len(self.running) <= spec.max_batch_size \
            and all(len(seq) + spec.num_speculative_tokens <= self.max_model_len for seq in self.running):
        n = 1 + spec.num_speculative_tokens
    while self.running and len(scheduled_seqs) < self.max_num_seqs:
        seq = self.running.popleft()
        while not self.block_manager.can_append(seq, n):
            ...                                     # 抢占，同现有逻辑
        else:
            seq.num_scheduled_tokens = n
            seq.is_prefill = False
            self.block_manager.may_append(seq, n)
            scheduled_seqs.append(seq)
```

- **本步是否推测是整个 batch 统一决定的**。一个 batch 内要么都推测，要么都不推测，保证 verify 的形状统一；
- **按 batch 大小关闭**：running 超过 `max_batch_size` 时走普通 decode，原有的 CUDA Graph 直接可用；
- **接近 `max_model_len` 时关闭**：位置 `len + K` 不能超出 `max_model_len`，否则 rotary 的 cos/sin 表和 CUDA Graph 中按 `max_model_len` 分配的 block_tables 列数都会越界。这种情况很少，整批退回普通 decode 最简单；
- **token 预算**：已经在 Config 中断言 `max_num_seqs × (1 + K) ≤ max_num_batched_tokens`，这里不用再检查。

### 9. ModelRunner

#### (1) 验证：`prepare_verify`

```
每条序列：input_ids = [last_token, d1, ..., dK]，positions = [len−1, ..., len−1+K]
          slot_mapping 覆盖这 K+1 个位置，context_lens = len + K
set_context(False, slot_mapping=..., context_lens=..., block_tables=..., q_len=K+1)
```

**补齐**（只有 n-gram 会出现）：草稿少于 K 个的序列仍然输入 K+1 个位置，补齐位置的 token 随便填：

- 补齐位置的 `slot_mapping` 设为 -1，`store_kvcache` 会跳过写入；
- `context_lens` 仍然是 `len + K`，否则右下角对齐的 causal mask 会错位；
- 补齐位置会读取 `[len + num_drafts, len + K)` 范围内的 cache。因为调度时已经对**整个 batch** 预留了 `len + K` 个位置，这些位置都在合法的块中，读到的内容无意义，但对应的输出在拒绝采样中被 `valid` 屏蔽。

#### (2) Attention：decode 分支按 `q_len` reshape

[context.py](../../nanovllm/utils/context.py) 中新增 `q_len: int = 1`。[attention.py](../../nanovllm/layers/attention.py) 的 decode 分支改为：

```python
else:    # decode / verify / 草稿模型
    o = flash_attn_with_kvcache(q.view(-1, context.q_len, self.num_heads, self.head_dim), k_cache, v_cache,
                                cache_seqlens=context.context_lens, block_table=context.block_tables,
                                softmax_scale=self.scale, causal=True)
    o = o.view(-1, self.num_heads, self.head_dim)
```

`flash_attn_with_kvcache` 的 causal mask 是右下角对齐的：第 j 个 query 只能看到前 `cache_seqlens − q_len + j + 1` 个 key，正好是它自己和它之前的位置。q_len = 1 时与现有行为相同，所以这一处改动同时覆盖了普通 decode（1）、验证（K+1）、草稿模型第 0 步（2）和后续步（1）四种情况。

最后的 `view` 是必须的：[qwen3.py](../../nanovllm/models/qwen3.py) 中 `o_proj(o.flatten(1, -1))` 要求 o 的形状是 `(tokens, H, D)`。

#### (3) LMHead 不用改

[embed_head.py](../../nanovllm/layers/embed_head.py) 的 `ParallelLMHead` 只在 `is_prefill` 时截取每条序列的最后一个位置；验证走的是 decode 路径，B·(K+1) 行都会输出 logits，正好是拒绝采样需要的。

#### (4) CUDA Graph

| 图 | 模型 | 每条序列的 Q 长度 | 用途 | 何时录制 |
|---|---|---|---|---|
| `graphs`（已有） | 大模型 | 1 | 普通 decode | 总是 |
| `verify_graphs` | 大模型 | K+1 | 验证 | 开启推测解码 |
| `draft_graphs[2]` | 草稿模型 | 2 | 草稿第 0 步 | method = draft_model |
| `draft_graphs[1]` | 草稿模型 | 1 | 草稿第 1..K−1 步 | method = draft_model |

现有的 `capture_cudagraph` 可以抽成 `capture(model, q_len)`，按 `graph_bs` 分档、为每组图分配 `max_bs × q_len` 大小的 `graph_vars`。`run_model` 根据 `context.q_len` 和模型选择对应的图。

所有图共用同一个内存池，并且先录 token 数最多的：verify 图（从大到小）→ 普通 decode 图 → 草稿模型的两组图。这样池的大小由最大的 verify 图决定。CUDA Graph 的录制机制、限制条件，以及推测解码为满足这些条件所做的设计，详见 [basic/cuda-graph.md](../basic/cuda-graph.md)。

#### (5) `run_spec`

```python
@torch.inference_mode()
def run_spec(self, seqs: list[Sequence]):
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    if self.spec.method == "ngram":
        draft_ids, num_drafts, draft_probs = self.ngram_proposer.propose(seqs)       # rank 0
    else:
        draft_ids, num_drafts, draft_probs = self.propose_draft_model(seqs, temperatures)
    input_ids, positions = self.prepare_verify(seqs, draft_ids, num_drafts)
    logits = self.run_model(input_ids, positions, False)
    reset_context()
    if self.rank != 0:
        return None
    return self.rejection_sampler(logits, temperatures, draft_ids, num_drafts, draft_probs)
```

`propose_draft_model` 循环 K 次：准备输入（第 0 次 Q=2，之后 Q=1）→ `run_draft_model`（走对应的草稿图）→ `sampler.sample_with_probs` 得到 token 和 q。

`run_spec` **不修改 Sequence**，只返回 token 列表，所有状态变更都在 `Scheduler.postprocess_spec` 中完成。

#### (6) Sampler

现有的 `Sampler.forward` 用 Gumbel-max 直接返回 token。草稿模型需要同时拿到 token 和分布 q，新增：

```python
def sample_with_probs(self, logits, temperatures):
    probs = torch.softmax(logits.float() / temperatures[:, None], -1)
    return torch.multinomial(probs, 1).squeeze(1), probs
```

### 10. 张量并行（TP > 1）

第一版在 Config 中断言 `tensor_parallel_size == 1`。架构上已经为 TP 留好了位置，跑通单卡后按下面的方式扩展：

- `run_spec` 通过 `call` 在所有 rank 上执行，大模型和草稿模型的前向在各 rank 上都正常参与 all-reduce；
- **草稿 token 只在 rank 0 上产生**（n-gram 在 rank 0 上查找；草稿模型的 logits 只 gather 到 rank 0 采样），每产生一批就用 `dist.broadcast` 发给其他 rank。草稿模型每一步一个 `[B]` 的 int64 张量，n-gram 一次 `[B, K]` 加 `[B]`，开销都很小；
- `num_drafts` 也一起广播，其他 rank 准备 `slot_mapping` 时要用；
- `Sequence.__getstate__`：草稿模型第 0 步需要最后 2 个 token。开启 draft_model 时，decode 阶段传 `token_ids[-2:]` 而不是只传 `last_token`。

### 11. 统计指标

推测解码的效果主要取决于接受率，调参时必须能看到它。Scheduler 中累计两个计数：

```
num_draft_tokens    += Σ num_drafts       （提议的草稿数）
num_accepted_tokens += Σ 接受数            （不含修正 / bonus token）
接受率 = num_accepted_tokens / num_draft_tokens
平均每步产出 = (num_accepted_tokens + 推测步数 × B) / (推测步数 × B)
```

`LLMEngine.generate` 的进度条中，decode 吞吐要按 `postprocess_spec` 返回的实际 token 数计算，而不是 `len(seqs)`；结束时打印接受率。

---

## 六、设计取舍：考虑过但没有采用的方案

| 方案 | 问题 | 本设计的做法 |
|---|---|---|
| **验证复用 prefill 的 varlen 路径** | 实现最简单，不需要改 attention；但 prefill 一律走 eager，验证步失去 CUDA Graph，省下的步数被 kernel 启动开销吃掉一部分；LMHead 还要新增"返回所有位置"的参数 | 走 decode 路径，`q_len = K+1`，单独录图；LMHead 不用改 |
| **草稿模型独立的一套块管理**（两个 free 队列、两张 block_table） | 两个池的块数相等、分配和释放同步进行，维护的是同一份信息；草稿池也拿不到 prefix cache | 共用 block_table，两个池按相同的块号寻址 |
| **记录草稿模型的 KV 进度，落后时先做一次草稿 prefill** | 需要额外的字段；只要 batch 中有一条序列落后，第 0 步就整体走 eager 的 varlen 路径 | 第 0 步固定 Q=2，形状规整 |
| **草稿 token 写入 `token_ids`，验证后回滚** | Sequence 中出现未确认的内容；需要 `pop` 操作；回滚分散在 ModelRunner 和 Scheduler 两处 | 草稿只存在于 ModelRunner 的局部张量中；`token_ids` 只包含已确认的内容 |
| **ModelRunner 直接修改 Sequence** | 状态被两处修改；worker 进程拿到的 Sequence 是每步重新反序列化的副本，修改它没有意义 | ModelRunner 只返回结果，Scheduler 统一修改 |
| **n-gram 在 LLMEngine / Scheduler 中调用，草稿作为参数传给 ModelRunner** | 两种方法的调用位置不同，step() 中需要分支 | 两种 propose 都在 `run_spec` 内部，step() 只区分"普通 decode / 推测步" |
| **按需截短草稿：空间不够时减少 K** | 一个 batch 内各序列的草稿数不同，verify 补齐时要读取未分配的块，需要额外处理 block_table 的填充值 | 整个 batch 统一预留 `len + K`，不够就抢占 |
| **验证后释放多余的块** | 下一步大概率马上又要分配；多一组释放逻辑 | 保留，由 `deallocate` 一并释放 |
| **Config 中平铺多个字段**（`speculative_model`、`num_speculative_tokens` 等） | 字段多了之后，哪个方法用哪些字段不清楚 | 集中在 `SpeculativeConfig` 中，按方法分别校验 |
| **每一步都传输完整的 `token_ids`** | 违背当前项目"decode 时只传 `last_token`"的设计，长序列更容易超出 1MB 的共享内存 | 只多传 1 个 token；草稿通过 `dist.broadcast` 传输 |

---

## 七、落地

### 1. 与现有机制的配合

| 机制 | 需要注意的地方 |
|---|---|
| **prefill 优先** | 推测解码只发生在 decode step；prefill 时草稿模型与大模型使用同样的输入一起计算 |
| **chunked prefill** | 每个 chunk 两个模型都要算。最后一个 chunk 完成后序列进入 running，下一步开始才可能推测 |
| **抢占** | 草稿从未写入 `token_ids`，重新 prefill 时不会带入未确认的内容；因为共用 block_table，两个池的块一起释放 |
| **prefix cache** | 依靠"先修正 `num_scheduled_tokens`，再调用 `hash_blocks`"保证被拒绝的 KV 永远不会登记 hash；命中的块在两个池中都有效 |
| **max_tokens / EOS** | 一步最多产出 K+1 个 token，追加时逐个检查，满足停止条件立即停止 |
| **max_model_len** | 任何序列 `len + K` 超出时，整批退回普通 decode |
| **CUDA Graph** | 关闭推测解码、或本步不推测时，走原有的 decode 图；`enforce_eager=True` 时所有新增的图都不录制 |

### 2. 文件改动清单

| 文件 | 改动 |
|---|---|
| **config.py** | 新增 `SpeculativeConfig`；`Config` 新增 `speculative_config` 字段，dict 转换与校验 |
| **新增 spec_decode/ngram_proposer.py** | `NgramProposer`：从长到短查找最近的一次匹配，输出 `draft_ids` / `num_drafts` |
| **新增 spec_decode/rejection_sampler.py** | `RejectionSampler`：统一处理 q 为 one-hot / 完整分布、草稿不足 K 个的情况 |
| **engine/block_manager.py** | `can_append(seq, n)` / `may_append(seq, n)`，n=1 时行为不变 |
| **engine/scheduler.py** | decode 分支决定 n；新增 `postprocess_spec`；抽出 `is_stop`；累计接受率 |
| **engine/sequence.py** | 不需要新字段；开启 draft_model 时 `__getstate__` 在 decode 阶段传最后 2 个 token（TP 扩展时） |
| **engine/model_runner.py** | 按配置创建 `NgramProposer` 或加载草稿模型；`allocate_kv_cache` 计入草稿模型并绑定 `draft_kv_cache`；prefill 时草稿模型一起前向；新增 `prepare_verify`、`propose_draft_model`、`run_spec`；`capture_cudagraph` 抽成按 `(model, q_len)` 录制 |
| **engine/llm_engine.py** | `step()` 按 `num_scheduled_tokens` 选择 `run` / `run_spec`；吞吐按实际 token 数统计；打印接受率 |
| **utils/context.py** | 新增 `q_len: int = 1` |
| **layers/attention.py** | decode 分支按 `q_len` reshape q 和 o |
| **layers/sampler.py** | 新增 `sample_with_probs` |
| **layers/embed_head.py** | 不用改 |
| **example.py / bench.py** | 新增推测解码相关的命令行参数 |

### 3. 推荐的实现顺序与验证方法

1. **配置与骨架**：`SpeculativeConfig`、`can_append(seq, n)` / `may_append(seq, n)`、Scheduler 的分支、`postprocess_spec`，`q_len` 改动。先让 propose 永远返回 `num_drafts = 0`：每步只产出 1 个 token，输出分布应该与关闭推测解码时完全一致。这一步验证了预留、补齐、后处理和 `hash_blocks` 的正确性；
2. **n-gram + 拒绝采样**：接入 `NgramProposer` 和 `RejectionSampler`，先用 `enforce_eager=True`。用重复性强的 prompt（例如"把下面这段代码原样输出"）检查接受率应该明显大于 0；
3. **草稿模型**：加载、显存规划、prefill 时一起计算、Q=2 的第 0 步。一个很好的检查方法是**让草稿模型等于大模型**：这时 p = q，接受率应该接近 100%，每步产出接近 K+1 个 token；
4. **边界测试**：用小 block_size 和小 KV 池，专门测试抢占、prefix cache 命中、`max_tokens` 和 EOS 出现在草稿中间、序列长度接近 `max_model_len` 的情况；
5. **CUDA Graph**：录制 verify 图和两组草稿图，对比开启前后的吞吐；
6. **确定默认值**：在 B = 1、8、16、32、64 上分别测试两种方法，确定 `max_batch_size` 和 K 的默认值；
7. **（可选）** TP > 1、按接受率动态调整 K、异步生成草稿（GPU 执行第 t 步时 CPU 准备第 t+1 步的 n-gram 草稿）、EAGLE / MTP 风格的草稿头（作为第三种 `method` 接入同一框架）。
