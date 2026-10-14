import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384                      # 一步最多计算的 token 数（所有序列之和）
                                                             #   prefill：Scheduler 按它装填，超出的部分用 chunked prefill 切开
                                                             #   decode：每条序列只算 1 个 token，实际不会触及这个上限
                                                             #   warmup 按它做一次最大规模的 prefill 测出激活峰值，剩余显存才分给 KV cache
                                                             #   → 调大能提高 prefill 吞吐，但激活占得多，KV cache 块变少
    max_num_seqs: int = 512                                  # 一步最多同时处理的序列数（prefill 和 decode 都受限）
                                                             #   decode 的 batch 大小上限就是它；CUDA Graph 按 min(它, 512) 录制
                                                             #   → 调大能提高 decode 吞吐，但需要更多 KV cache，容易触发抢占
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
