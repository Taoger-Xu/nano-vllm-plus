from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """一次前向的批次元数据，以全局变量的形式传给模型内部，不必一层层作为参数传递。

    由 ModelRunner 在前向之前通过 set_context 设置（prepare_prefill / prepare_decode），
    由 Attention（选择 flash-attn 接口、写 KV cache）和 ParallelLMHead（prefill 时只取最后一个位置）读取。

    例 1：prefill，block_size = 4。A 命中 4 个 token 的前缀、本步算 6 个；B 是全新请求、算 5 个
        is_prefill   = True
        cu_seqlens_q = [0, 6, 11]        A 的 query 是第 0~5 个 token，B 是第 6~10 个
        cu_seqlens_k = [0, 10, 15]       A 要看 10 个 key（4 个前缀 + 6 个新的），B 看 5 个
        max_seqlen_q = 6, max_seqlen_k = 10
        slot_mapping = [32 33 34 35 20 21 | 24 25 26 27 4]    11 个新 token 的 KV 各写到哪
        block_tables = [[3, 8, 5], [6, 1, -1]]                有前缀命中才需要
        context_lens = None

    例 2：decode，block_size = 256。三条序列，每条 1 个新 token
        is_prefill   = False
        slot_mapping = [2347, 521, 2816]                      3 个新 token 的 KV 各写到哪
        context_lens = [300, 10, 513]                         每条序列要读多少个 KV
        block_tables = [[5, 9, -1], [2, -1, -1], [7, 3, 11]]
        cu_seqlens_* / max_seqlen_* 不使用

    prefill 每条序列的 token 数不同，用 cu_seqlens_* 描述序列边界，走 flash_attn_varlen_func；
    decode 每条序列恰好 1 个 token，第 i 行就是第 i 条序列，用 context_lens 说明长度，走 flash_attn_with_kvcache。
    decode 用到的字段全是张量、没有 Python 标量，所以可以录 CUDA Graph（录图时这些张量是 graph_vars 的切片）。
    """
    is_prefill: bool = False                         # 本步是 prefill 还是 decode；Attention 和 LMHead 据此分支
    cu_seqlens_q: torch.Tensor | None = None         # prefill：每条序列的 query 在一维拼接数组中的边界（前缀和）
    cu_seqlens_k: torch.Tensor | None = None         # prefill：每条序列的 key 的边界，包含已缓存的前缀
    max_seqlen_q: int = 0                            # prefill：本批最长的 query 长度（Python int，flash-attn 用来规划 kernel）
    max_seqlen_k: int = 0                            # prefill：本批最长的 key 长度
    slot_mapping: torch.Tensor | None = None         # 两者：本步每个 token 的 KV 写到 cache 的第几个 slot；-1 表示不写
    context_lens: torch.Tensor | None = None         # decode：每条序列当前的总长度，即 attention 要读多少个 KV
    block_tables: torch.Tensor | None = None         # 每条序列的逻辑块 → 物理块；decode 总是有，prefill 只在 K 比 Q 长时有

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    # 每次前向之前由 ModelRunner 调用，必须与 reset_context 成对使用
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    # 前向结束后清空，避免下一次前向误用上一批的元数据
    global _CONTEXT
    _CONTEXT = Context()
