from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()                                         # 在 Scheduler.waiting 中：还没 prefill 完，或被抢占
    RUNNING = auto()                                         # 在 Scheduler.running 中：prefill 已完成，逐步 decode
    FINISHED = auto()                                        # 遇到 EOS 或达到 max_tokens，块已归还


class Sequence:
    block_size = 256                                         # 类属性，LLMEngine 启动时改成 config.kvcache_block_size
    counter = count()                                        # 全局自增计数器，用来生成 seq_id

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)                 # 唯一标识，generate() 用它把输出对应回输入顺序
        self.status = SequenceStatus.WAITING                 # 新请求先进入 waiting 队列
        self.token_ids = copy(token_ids)                     # prompt + 已生成的 token（只包含已确认的内容）；拷贝一份，避免改到调用者的列表
        self.last_token = token_ids[-1]                      # 最后一个 token：decode 时的输入，跨进程时只需传它
        self.num_tokens = len(self.token_ids)                # 当前总长度 = len(token_ids)，单独存一份是因为 worker 端不一定有完整 token_ids
        self.num_prompt_tokens = len(token_ids)              # prompt 长度，保持不变
        self.num_cached_tokens = 0                           # KV 已经在 cache 中的 token 数：prefix cache 命中 + 之前各步算完的部分
                                                             #   本步从这个位置开始计算
        self.num_scheduled_tokens = 0                        # 本步要计算的 token 数：prefill 时为一个 chunk 的长度，decode 时为 1
                                                             #   本步计算范围 = [num_cached_tokens, num_cached_tokens + num_scheduled_tokens)
        self.is_prefill = True                               # 是否还处于 prefill 阶段；决定跨进程时传完整 token_ids 还是只传 last_token
        self.block_table = []                                # 逻辑块 i → 物理块号，由 BlockManager 分配；为空表示还没分配（新请求或被抢占）
        self.temperature = sampling_params.temperature       # 采样温度
        self.max_tokens = sampling_params.max_tokens         # 最多生成多少个 token
        self.ignore_eos = sampling_params.ignore_eos         # True 时遇到 EOS 也不停止（benchmark 用）

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
        return (self.num_tokens + self.block_size - 1) // self.block_size   # 需要的块数，向上取整

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size    # 最后一块中有几个 token（1 ~ block_size）

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]    # 第 i 个逻辑块的 token，用于计算哈希

    def append_token(self, token_id: int):
        # postprocess 中调用：把本步采样出的 token 追加到序列。它的 KV 要到下一步 decode 时才写入 cache
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # TP > 1 时 rank 0 把 seqs pickle 到共享内存发给其他 rank，这里控制传输的数据量
        # prefill 需要完整的 token_ids 作为输入；decode 只需要 last_token，不传整个历史
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        # worker 端还原：只恢复 ModelRunner 准备输入时用到的字段（没有 status、sampling 参数等）
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state                      # prefill：拿到完整 token_ids
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []                              # decode：只有 last_token，token_ids 为空
            self.last_token = last_state
