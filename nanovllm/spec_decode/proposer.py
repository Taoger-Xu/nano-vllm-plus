from abc import ABC, abstractmethod

from nanovllm.engine.sequence import Sequence


class Proposer(ABC):
    """统一的 speculative decoding 草稿生成接口。"""

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    @abstractmethod
    def propose(self, seqs: list[Sequence], temperatures=None):
        """返回 ``draft_ids``、``num_drafts`` 和 ``draft_probs``。"""
        raise NotImplementedError
