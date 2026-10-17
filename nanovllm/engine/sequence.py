from copy import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    # 在 Scheduler.waiting 中：还没 prefill 完，或被抢占
    WAITING = auto()
    # 在 Scheduler.running 中：prefill 已完成，逐步 decode
    RUNNING = auto()
    # 遇到 EOS 或达到 max_tokens，块已归还
    FINISHED = auto()


@dataclass(slots=True)
class KVState:
    """一条序列在某一个模型（大模型或草稿模型）的 KV cache 中的状态。

    每个模型各有一份：块各自分配，进度各自推进。本步计算范围为
    [num_cached_tokens, num_cached_tokens + num_scheduled_tokens)。
    """
    # KV 已经写入 cache 的 token 数：prefix cache 命中 + 之前各步算完的部分；
    # 本步从这里开始计算
    num_cached_tokens: int = 0
    # 本步要计算的 token 数：prefill 时为一个 chunk 的长度，decode 时为 1（推测步为 1 + K）
    num_scheduled_tokens: int = 0
    # 逻辑块 i → 物理块号，由对应模型的 BlockManager 分配；为空表示还没分配（新请求或被抢占）
    block_table: list[int] = field(default_factory=list)


class Sequence:
    # 类属性，LLMEngine 启动时改成 config.kvcache_block_size
    block_size = 256
    # 全局自增计数器，用来生成 seq_id
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """创建一个新请求的序列状态。

        Args:
            token_ids (list[int]): prompt 的 token；会拷贝一份，避免改到调用者的列表。
            sampling_params (SamplingParams): 采样参数。
        """
        # 唯一标识，generate() 用它把输出对应回输入顺序
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        # prompt + 已生成的 token，只包含已确认的内容
        self.token_ids = copy(token_ids)
        # decode 时的输入，跨进程时只需传它
        self.last_token = token_ids[-1]
        # 单独存一份长度：worker 端不一定有完整的 token_ids
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        # 大模型的 KV 状态；block_table / num_cached_tokens / num_scheduled_tokens 是它的转发属性
        self.target_kv = KVState()
        # 草稿模型的 KV 状态；不开启草稿模型时保持为空
        self.draft_kv = KVState()
        # 是否还处于 prefill 阶段；决定跨进程时传完整 token_ids 还是只传 last_token
        self.is_prefill = True
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        # True 时遇到 EOS 也不停止（benchmark 用）
        self.ignore_eos = sampling_params.ignore_eos

    def kv(self, role: str) -> KVState:
        """按角色取 KV 状态，供 BlockManager / ModelRunner 按模型区分读写。

        Args:
            role (str): "target" 为大模型，"draft" 为草稿模型。

        Returns:
            KVState: 对应模型的 KV 状态。
        """
        if role == "target":
            return self.target_kv
        assert role == "draft", role
        return self.draft_kv

    # 以下三个属性转发到 target_kv，保持原有代码（只有一个模型时）不用改动
    @property
    def block_table(self) -> list[int]:
        return self.target_kv.block_table

    @block_table.setter
    def block_table(self, value: list[int]):
        self.target_kv.block_table = value

    @property
    def num_cached_tokens(self) -> int:
        return self.target_kv.num_cached_tokens

    @num_cached_tokens.setter
    def num_cached_tokens(self, value: int):
        self.target_kv.num_cached_tokens = value

    @property
    def num_scheduled_tokens(self) -> int:
        return self.target_kv.num_scheduled_tokens

    @num_scheduled_tokens.setter
    def num_scheduled_tokens(self, value: int):
        self.target_kv.num_scheduled_tokens = value

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """覆盖当前全部 token 需要的块数（向上取整）。"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一块中有几个 token（1 ~ block_size）。"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """第 i 个逻辑块的 token，用于计算哈希。

        Args:
            i (int): 逻辑块号。

        Returns:
            list[int]: 该块的 token。
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """把本步采样出的 token 追加到序列，由 postprocess 调用。它的 KV 要到下一步 decode 时才写入 cache。

        Args:
            token_id (int): 新 token。
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """TP > 1 时 rank 0 把 seqs pickle 到共享内存发给其他 rank，这里控制传输的数据量。

        prefill 需要完整的 token_ids 作为输入；decode 只需要 last_token，不传整个历史。
        两个模型的 KV 状态都要传：worker 需要各自的 block_table 和进度来构造输入。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        target, draft = self.target_kv, self.draft_kv
        return (self.num_tokens, self.num_prompt_tokens,
                target.num_cached_tokens, target.num_scheduled_tokens, target.block_table,
                draft.num_cached_tokens, draft.num_scheduled_tokens, draft.block_table,
                last_state)

    def __setstate__(self, state):
        """worker 端还原：只恢复 ModelRunner 准备输入时用到的字段（没有 status、sampling 参数等）。"""
        (self.num_tokens, self.num_prompt_tokens,
         target_cached, target_scheduled, target_table,
         draft_cached, draft_scheduled, draft_table,
         last_state) = state
        self.target_kv = KVState(target_cached, target_scheduled, target_table)
        self.draft_kv = KVState(draft_cached, draft_scheduled, draft_table)
        if isinstance(last_state, list):
            # prefill：拿到完整 token_ids
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # decode：只有 last_token，token_ids 为空
            self.token_ids = []
            self.last_token = last_state
