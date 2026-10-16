import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from nanovllm.engine.sequence import Sequence
from nanovllm.spec_decode.proposer import Proposer


class NgramProposer(Proposer):
    """从每条序列的已确认 token 中查找重复 n-gram。"""

    def __init__(self, k: int, n_min: int, n_max: int):
        super().__init__(k)
        if not 1 <= n_min <= n_max:
            raise ValueError("n_min must satisfy 1 <= n_min <= n_max")
        self.n_min = n_min
        self.n_max = n_max

    def lookup(self, tokens: np.ndarray) -> np.ndarray:
        """查找最后一个 n-gram 最近一次出现位置之后的 token。"""
        for n in range(self.n_max, self.n_min - 1, -1):
            if tokens.size <= n:
                continue
            windows = sliding_window_view(tokens[:-1], n)
            matches = np.flatnonzero((windows == tokens[-n:]).all(axis=1))
            if matches.size:
                start = int(matches[-1]) + n
                return tokens[start:start + self.k]
        return tokens[:0]

    @torch.inference_mode()
    def propose(self, seqs: list[Sequence], temperatures=None):
        """生成 ``[B, K]`` 草稿；n-gram 使用 one-hot 分布，故返回 None。"""
        draft_ids = torch.zeros(len(seqs), self.k, dtype=torch.int64)
        num_drafts = torch.zeros(len(seqs), dtype=torch.int64)
        for row, seq in enumerate(seqs):
            draft = self.lookup(np.asarray(seq.token_ids, dtype=np.int64))
            length = min(len(draft), self.k)
            if length:
                draft_ids[row, :length] = torch.from_numpy(draft[:length])
            num_drafts[row] = length
        return draft_ids.cuda(non_blocking=True), num_drafts.cuda(non_blocking=True), None
