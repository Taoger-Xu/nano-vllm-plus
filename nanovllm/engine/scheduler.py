from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        """
        创建调度器：维护 waiting / running 两个队列维护序列的调度
        和 BlockManager负责管理KV cache
        开启草稿模型时候两个 BlockManager 同进同退：
        准入、预留要两边都放得下，抢占、结束时两边一起释放。

        Args:
            表示 KV cache 池中一共有多少个物理块
        """
        # 每步 batch 的序列数上限
        self.max_num_seqs = config.max_num_seqs
        # 每步 batch 的 token 数上限（只在 prefill 中检查）
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size,
                                          enable_prefix_cache=config.enable_prefix_cache)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # None 表示未开启推测解码
        self.spec = config.speculative_config
        self.num_spec_tokens = self.spec.num_speculative_tokens if self.spec else 0
        self.max_model_len = config.max_model_len
        # 临时开关：run_spec / postprocess_spec 完成之前保持 False
        self.spec_enabled = False
        # 草稿模型的 BlockManager；草稿模型不做 prefix cache。块数由 ModelRunner.allocate_kv_cache 写入，块号空间各自独立
        self.draft_block_manager = None
        if self.spec is not None and self.spec.method == "draft_model":
            self.draft_block_manager = BlockManager(config.num_draft_kvcache_blocks, config.kvcache_block_size,
                                                    role="draft", enable_prefix_cache=False)


    def is_finished(self):
        """所有请求是否都已完成。

        Returns:
            bool: waiting 和 running 都为空时为 True。
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """接收一个新请求，放进 waiting 队尾。此时 block_table 为空、num_cached_tokens = 0。

        Args:
            seq (Sequence): 由 LLMEngine.add_request 创建的序列。
        """
        self.waiting.append(seq)

    def _block_managers(self) -> list[BlockManager]:
        """本引擎使用的所有 BlockManager：大模型一个，开启草稿模型时再加一个。"""
        if self.draft_block_manager is None:
            return [self.block_manager]
        return [self.block_manager, self.draft_block_manager]

    def _decode_num_tokens(self) -> int:
        """决定本步 decode 每条序列计算多少个 token，整批统一。

        Returns:
            int: 1 + K 为推测步，1 为普通 decode。
        """
        if not (self.spec_enabled and self.spec is not None):
            return 1
        # batch 太大，推测不划算
        if len(self.running) > self.spec.max_batch_size:
            return 1
        # 写入位置会超出 max_model_len
        if any(len(seq) + self.num_spec_tokens > self.max_model_len for seq in self.running):
            return 1
        return 1 + self.num_spec_tokens

    def _draft_num_tokens(self, seq: Sequence, n: int) -> int:
        """本步草稿模型要写入多少个 token 的 KV。

        推测步中草稿模型要写到位置 len(seq) - 2 + K（输入 last_token、d1 .. d_{K-1}），
        并先补上此前落后的部分，所以从 draft_kv.num_cached_tokens 一直算到那里；
        不推测时草稿模型不运行，允许落后，下次推测时再补。

        Args:
            seq (Sequence): decode 中的序列。
            n (int): 大模型本步计算的 token 数。

        Returns:
            int: 草稿模型本步要写入的 token 数；不推测时为 0。
        """
        if n == 1:
            return 0
        return len(seq) - 1 + (n - 1) - seq.draft_kv.num_cached_tokens

    def schedule(self) -> tuple[list[Sequence], bool]:
        """选出本步要计算的序列。prefill 优先；没有可 prefill 的序列时才做 decode。

        prefill：从 waiting 队首依次装入，受序列数和 token 预算限制，只允许本批第一条被切分（chunked prefill）。
        decode：先整批决定每条序列算几个 token，空间不够时从 running 队尾抢占。
        开启草稿模型时，两个模型的块都要放得下才调度。

        Returns:
            tuple[list[Sequence], bool]: 本步的序列，以及是否为 prefill。
        """
        scheduled_seqs = []
        num_batched_tokens = 0
        draft_bm = self.draft_block_manager

        # prefill：限制一是序列数，限制二是本步剩余的 token 预算
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 第一次调度（新请求，或被抢占后 block_table 已清空）：查 prefix cache，检查空闲块够不够
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                # 草稿模型也要放得下；它不做 prefix cache，返回值恒为 0 或 -1
                if draft_bm is not None and draft_bm.can_allocate(seq) == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # chunked prefill 的后续块：块已分配，接着上次的进度算
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 放不下且不是本批第一条 → 留到下一步，不切分
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                if draft_bm is not None:
                    draft_bm.allocate(seq, 0)
            # 本批第一条放不下时只取 remaining 个 → chunked prefill
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            if draft_bm is not None:
                # 开启草稿模型时 prefix cache 已关闭，两个模型的 prefill 进度始终相同，算同一段
                assert seq.draft_kv.num_cached_tokens == seq.num_cached_tokens
                seq.draft_kv.num_scheduled_tokens = seq.num_scheduled_tokens
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode：只受序列数限制；Config 已断言 max_num_seqs × (1 + K) ≤ max_num_batched_tokens
        n = self._decode_num_tokens()
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            n_draft = self._draft_num_tokens(seq, n) if draft_bm is not None else 0
            # 空间不够（任一模型）：从 running 队尾抢占，直到放得下；队里没有别人了就抢占自己
            while not (self.block_manager.can_append(seq, n)
                       and (draft_bm is None or draft_bm.can_append(seq, n_draft))):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = n
                seq.is_prefill = False
                self.block_manager.may_append(seq, n)
                if draft_bm is not None:
                    seq.draft_kv.num_scheduled_tokens = n_draft
                    draft_bm.may_append(seq, n_draft)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占一条序列：释放它在所有模型中的块，放回 waiting 队首，之后重新 prefill。

        Args:
            seq (Sequence): 被抢占的序列。
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        for bm in self._block_managers():
            bm.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """一步结束后更新序列状态：登记写满的块、推进进度、追加新 token、处理结束的序列。

        两个模型各自推进进度：prefill 时进度相同；不推测的 decode 步草稿模型没有运行，进度不动（落后）。

        Args:
            seqs (list[Sequence]): 本步计算的序列。
            token_ids (list[int]): 每条序列采样出的 token。
            is_prefill (bool): 本步是否为 prefill；prefill 中间的 chunk 不追加 token。
        """
        for seq, token_id in zip(seqs, token_ids):
            for bm in self._block_managers():
                kv = seq.kv(bm.role)
                bm.hash_blocks(seq)
                kv.num_cached_tokens += kv.num_scheduled_tokens
                kv.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                for bm in self._block_managers():
                    bm.deallocate(seq)
                self.running.remove(seq)
