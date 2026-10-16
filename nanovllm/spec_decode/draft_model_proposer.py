import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.spec_decode.proposer import Proposer


class DraftModelProposer(Proposer):
    """通过 ModelRunner 的草稿模型步骤生成 token 和完整 q 分布。

    ModelRunner 需要提供 ``run_draft_step(seqs, input_ids, positions)``，
    返回 ``(next_ids, probs)``，其中两个张量的形状分别为 ``[B]`` 和
    ``[B, V]``。这样 proposer 不需要复制 KV cache 和 CUDA Graph 管理逻辑。
    """

    def __init__(self, model_runner, k: int):
        super().__init__(k)
        self.model_runner = model_runner

    @torch.inference_mode()
    def propose(self, seqs: list[Sequence], temperatures):
        """循环 K 步生成 ``draft_ids [B,K]`` 和 ``draft_probs [B,K,V]``。"""
        if temperatures is None:
            raise ValueError("temperatures are required for draft_model")
        if not hasattr(self.model_runner, "run_draft_step"):
            raise NotImplementedError(
                "ModelRunner.run_draft_step is required for draft-model proposing"
            )

        batch_size = len(seqs)
        draft_ids = []
        draft_probs = []
        input_ids, positions = self.model_runner.prepare_draft_decode(seqs)

        for _ in range(self.k):
            token_ids, probs = self.model_runner.run_draft_step(
                seqs, input_ids, positions, temperatures
            )
            draft_ids.append(token_ids)
            draft_probs.append(probs)
            input_ids = token_ids
            positions = positions + 1

        draft_ids = torch.stack(draft_ids, dim=1)
        draft_probs = torch.stack(draft_probs, dim=1)
        num_drafts = torch.full(
            (batch_size,), self.k, dtype=torch.int64, device=draft_ids.device
        )
        return draft_ids, num_drafts, draft_probs
