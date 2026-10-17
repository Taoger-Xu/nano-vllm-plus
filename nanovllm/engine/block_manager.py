from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """一个物理 KV 块的元数据（账本）。真正的 K/V 存在 kv_cache[..., block_id, ...] 中。"""

    def __init__(self, block_id):
        """创建一个空闲块的元数据。

        Args:
            block_id (int): 物理块号，对应 kv_cache 中的下标。
        """
        self.block_id = block_id
        # 正在使用它的序列数；0 = 空闲
        self.ref_count = 0
        # 链式哈希；-1 = 未登记（没写满或已作废）
        self.hash = -1
        # 块内 token，查找时防哈希碰撞
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """块写满后登记它的身份，由 hash_blocks 调用。

        Args:
            hash (int): 从开头到本块的链式哈希。
            token_ids (list[int]): 块内的 token，查找时用来防止哈希碰撞。
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """把块重新初始化，交给一条新序列使用。

        只在 _allocate_block 中调用：此时块正被交给一条新序列，所以使用者数是 1，而不是 0；
        旧内容即将被覆盖，旧身份作废。
        """
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        """创建分页 KV cache 的块管理器，所有块初始为空闲。

        Args:
            num_blocks (int): 物理块总数，由 ModelRunner.allocate_kv_cache 按显存计算。
            block_size (int): 每块容纳的 token 数。
        """
        self.block_size = block_size
        # 所有块的元数据，下标 = 物理块号
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 链式哈希 → 物理块号；释放时不删除（惰性淘汰），所以登记的块可能在 free 中（可复活）
        self.hash_to_block_id: dict[int, int] = dict()
        # ref_count == 0 的块，FIFO：popleft 分配，append 释放
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # ref_count > 0 的块，与 free 互补
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算一个块的链式哈希：h_i = hash(h_{i-1} + 第 i 块的 token)。

        h_i 相同 ⇔ 从开头到第 i 块的整个前缀相同（KV 依赖前面所有 token，只比较本块内容是不够的）。

        Args:
            token_ids (list[int]): 本块的 token。
            prefix (int): 前一块的哈希；第 0 块没有前缀，传 -1。

        Returns:
            int: 本块的链式哈希。
        """
        h = xxhash.xxh64()
        # ① 先混入前一块的哈希，② 再混入本块的 token
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """从空闲队列头部取出一块分配给新序列；若它还带着旧的 hash 索引，此时才删除（惰性淘汰）。

        Returns:
            int: 分配到的物理块号。
        """
        # 取最早释放的块（近似 LRU）
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # == block_id：防止误删指向另一个同前缀块的索引
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """把引用计数已降为 0 的块移回空闲队列尾部。

        不清 hash / token_ids：KV 还在显存里，在被 _allocate_block 取走之前仍可被命中。

        Args:
            block_id (int): 要释放的物理块号。
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """只读检查：能命中多少块 prefix cache，以及空闲块够不够放下整条序列。

        不修改任何状态，调度器放弃时无需回滚。只在 seq.block_table 为空时由 Scheduler 调用
        （新请求、被抢占后重新调度）。

        Args:
            seq (Sequence): 待分配的序列。

        Returns:
            int: -1 表示空闲块不够；>= 0 表示命中的块数，原样传给 allocate。
        """
        h = -1
        num_cached_blocks = 0
        # ① 先假设所有块都要从 free 中取
        num_new_blocks = seq.num_blocks
        # ② 不匹配最后一块：保证至少算一段，才能拿到 logits
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            # ③ h 代表从开头到第 i 块的整个前缀
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 未命中或哈希碰撞 → 停止，前缀匹配必须连续
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # ④ 命中且在使用中 → 共享，不占 free；在 free 中 → 复活也要占 free
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        # ⑤ 空闲块不够 → 请求继续等待
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """为整条序列分配块：前 num_cached_blocks 块复用 prefix cache，其余新分配。

        完全信任 can_allocate 的返回值：两次调用之间 BlockManager 不能被修改。

        Args:
            seq (Sequence): 待分配的序列，block_table 必须为空。
            num_cached_blocks (int): can_allocate 返回的命中块数。
        """
        assert not seq.block_table
        h = -1
        # ① 先处理命中的块，再分配新块：否则 popleft 可能取走并 reset 一个即将复活的缓存块
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            # can_allocate 已保证命中，直接索引
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                # ② 别的序列正在使用 → 共享
                block.ref_count += 1
            else:
                # ③ 在 free 中但 KV 还在 → 复活
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # ④ 未命中的块：从 free 队头取新块
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        # ⑤ 本步从哪个 token 开始计算
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """序列结束或被抢占时，归还它对所有块的引用，并清空 block_table 和 num_cached_tokens。

        Args:
            seq (Sequence): 要释放的序列。
        """
        # ① 逆序：尾部块先进 free、先被覆盖，常被共享的开头块留存最久
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            # ② 共享块只减引用
            block.ref_count -= 1
            # ③ 没人用了 → 回到 free，hash 保留（可复活）
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        # ④ 清空后重新调度会再走 can_allocate，可能命中自己之前的块
        seq.block_table.clear()

    def _num_new_blocks(self, seq: Sequence, n: int) -> int:
        """计算本步还需要新分配多少块。

        本步写入 [num_cached_tokens, num_cached_tokens + n)，block_table 需覆盖到末尾；
        已有的块（含上一个推测步多留的）不重复分配。

        Args:
            seq (Sequence): decode 中的序列。
            n (int): 本步要写入 KV 的 token 数。

        Returns:
            int: 需要新分配的块数，不小于 0。
        """
        needed = (seq.num_cached_tokens + n + self.block_size - 1) // self.block_size
        return max(0, needed - len(seq.block_table))

    def can_append(self, seq: Sequence, n: int = 1) -> bool:
        """decode 前的只读检查：空闲块够不够本步写入 n 个 token。不够时 Scheduler 触发抢占。

        Args:
            seq (Sequence): decode 中的序列。
            n (int): 本步要写入 KV 的 token 数；1 为普通 decode，1 + K 为推测步。

        Returns:
            bool: 空闲块是否足够。
        """
        return len(self.free_block_ids) >= self._num_new_blocks(seq, n)

    def may_append(self, seq: Sequence, n: int = 1):
        """can_append 通过后调用，为本步补齐缺的块。

        Args:
            seq (Sequence): decode 中的序列。
            n (int): 本步要写入 KV 的 token 数，与 can_append 传入的相同。
        """
        for _ in range(self._num_new_blocks(seq, n)):
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """把本步中被"写满"的块登记到 hash_to_block_id，之后的请求才能命中它们。

        每步 postprocess 中调用（prefill 和 decode 都会经过），此时本步的 KV 已经写入 cache。
        本步计算的范围是 [num_cached_tokens, num_cached_tokens + num_scheduled_tokens)。

        Args:
            seq (Sequence): 本步参与计算的序列，num_scheduled_tokens 必须是本步已确认的 token 数。
        """
        # ① 本步开始时所在的块（可能已写了一部分，尚未登记）
        start = seq.num_cached_tokens // self.block_size
        # ② 向下取整：只登记写满的块
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        # 本步没有写满任何块（大多数 decode 步）
        if start == end: return
        # ③ 链式哈希的起点：前一块早已登记
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            # ④ 块记下自己的身份（淘汰时要用）
            block.update(h, token_ids)
            # ⑤ 登记全局索引；同前缀已登记时覆盖为这个块
            self.hash_to_block_id[h] = block.block_id
