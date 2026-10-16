import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    # 词表并行：embedding 表按词表（行）切，rank i 存 token id ∈ [vocab_start_idx, vocab_end_idx) 的向量
    # 词表很大（Qwen3 为 151936 × hidden），切开能省下可观的显存

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)   # 词表的第 tp_rank 段
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # 输入 x 是 token id，每个 rank 都有完整的一份（各 rank 根据控制面传来的 seqs 自己构造）
        # 例：TP=2，vocab=151936，token 100000 归 rank 1（[75968, 151936)）：
        #   rank 0：mask=False，查第 0 行后清零 → 全 0；rank 1：查本地第 24032 行 → 真实向量
        #   all-reduce 求和后，每个 rank 都得到完整的 [tokens, hidden]
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)   # 哪些 token 归本 rank 管
            x = mask * (x - self.vocab_start_idx)                    # 转成本地下标；不归自己管的变成 0（随便查一行，后面清零）
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y                                # 不归自己管的行清零
            dist.all_reduce(y)                                       # 每个 token 恰好一个 rank 给出非零向量
        return y


class ParallelLMHead(VocabParallelEmbedding):
    # lm_head 与 embedding 按同样的方式切词表，本质是一个列并行的线性层：rank i 算出词表第 i 段的 logits
    # tie_word_embeddings 时直接共享 embedding 的权重，两者切法相同，每个 rank 上的这一片正好对应

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1              # prefill 只需要每条序列最后一个位置，先缩小再做矩阵乘
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)                            # [序列数, vocab/N]
        if self.tp_size > 1:
            # 用 gather 而不是 all-gather：只有 rank 0 采样，其他 rank 不需要完整 logits，返回 None
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None   # 按词表顺序拼回 [序列数, vocab]
        return logits
