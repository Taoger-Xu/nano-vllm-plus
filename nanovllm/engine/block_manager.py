from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    # 一个物理 KV 块的元数据（账本）。真正的 K/V 存在 kv_cache[..., block_id, ...] 中

    def __init__(self, block_id):
        self.block_id = block_id                                     # 物理块号，对应 kv_cache 中的下标
        self.ref_count = 0                                           # 正在使用它的序列数；0 = 空闲
        self.hash = -1                                               # 链式哈希；-1 = 未登记（没写满，或内容已作废）
        self.token_ids = []                                          # 块内 token，查找时用来防止哈希碰撞

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash                                             # 块写满后由 hash_blocks 调用，登记身份
        self.token_ids = token_ids

    def reset(self):
        # 只在 _allocate_block 中调用：块正被交给一条新序列，所以使用者数是 1，而不是 0
        self.ref_count = 1
        self.hash = -1                                               # 旧内容即将被覆盖，旧身份作废
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size                                 # 每块容纳的 token 数
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]   # 所有块的元数据，下标 = 物理块号，长度固定
        self.hash_to_block_id: dict[int, int] = dict()               # 链式哈希 → 物理块号；块释放时不删除（惰性淘汰），
                                                                     #   所以登记的块可能在 used 中，也可能在 free 中（可复活）
        self.free_block_ids: deque[int] = deque(range(num_blocks))   # ref_count == 0 的块，FIFO：popleft 分配，append 释放
        self.used_block_ids: set[int] = set()                        # ref_count > 0 的块；与 free 互斥且互补

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 链式哈希：h_i = hash(h_{i-1} + 第 i 块的 token)，所以 h_i 相同 ⇔ 从开头到第 i 块的整个前缀相同
        # （KV 依赖前面所有 token，只比较本块内容是不够的）
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))                   # ① 先混入前一块的哈希；第 0 块没有前缀
        h.update(np.array(token_ids).tobytes())                      # ② 再混入本块的 token
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()                     # 取最早释放的块（近似 LRU）
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]                    # 惰性淘汰：被重新分配时才删除旧索引；
                                                                     #   == block_id 防止误删指向另一个同前缀块的索引
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        # 只把块移回 free 队尾，不清 hash / token_ids：KV 还在显存里，在被 _allocate_block 取走之前仍可被命中
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        # 只读的检查：能命中多少块、空闲块够不够。不修改任何状态，调度器放弃时无需回滚
        # 只在 seq.block_table 为空时由 Scheduler 调用（新请求、被抢占后重新调度）
        # 返回 -1 = 放不下；>= 0 = 命中的块数，原样传给 allocate
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks                              # ① 先假设所有块都要从 free 中取
        for i in range(seq.num_blocks - 1):                          # ② 不匹配最后一块：保证至少算一段，才能拿到 logits
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)                      # ③ 链式哈希：h 代表从开头到第 i 块的整个前缀
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break                                                #    未命中或哈希碰撞 → 停止，前缀匹配必须连续
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1                                  # ④ 命中且正在使用 → 只需 ref_count + 1，不占 free
                                                                     #    命中但在 free 中 → 复活也要从 free 取，不减
        if len(self.free_block_ids) < num_new_blocks:
            return -1                                                # ⑤ 空闲块不够 → 调度器 break，请求继续等待
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        # 真正修改状态的执行。完全信任 can_allocate 的返回值：两次调用之间 BlockManager 不能被修改
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):                           # ① 先处理命中的块（必须在分配新块之前，
            token_ids = seq.block(i)                                 #    否则 popleft 可能取走并 reset 一个即将复活的缓存块）
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]                      #    直接索引：can_allocate 已保证命中
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1                                 # ② 别的序列正在使用 → 共享
            else:
                block.ref_count = 1                                  # ③ 在 free 中但 KV 还在 → 复活
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())           # ④ 未命中的块：从 free 队头取新块
        seq.num_cached_tokens = num_cached_blocks * self.block_size  # ⑤ 告诉 Scheduler / ModelRunner 从哪个 token 开始计算

    def deallocate(self, seq: Sequence):
        # 序列结束或被抢占时调用：归还这条序列对所有块的引用
        for block_id in reversed(seq.block_table):                   # ① 逆序：尾部块先进入 free 队列、先被覆盖，
            block = self.blocks[block_id]                            #    最常被共享的开头块（system prompt）留存最久
            block.ref_count -= 1                                     # ② 共享块只减引用，别的序列还在用
            if block.ref_count == 0:
                self._deallocate_block(block_id)                     # ③ 没人用了 → 回到 free，但 hash 保留（可复活）
        seq.num_cached_tokens = 0
        seq.block_table.clear()                                      # ④ 清空后 block_table 为空 → 被抢占的序列重新调度时
                                                                     #    会再走 can_allocate，可能命中自己之前算过的块

    def can_append(self, seq: Sequence) -> bool:
        # decode 前的只读检查：这一步需要的新块（0 或 1 个）够不够。不够时 Scheduler 触发抢占
        # len(seq) 已包含上一步采样出的 last_token，本步要写入它的 KV，位置是 len(seq) - 1
        # len(seq) % block_size == 1 ⇔ 这个位置是新块的第一个 token → 需要 1 个新块（bool 当作 0/1 比较）
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # decode 的执行：can_append 通过后调用，必要时为 last_token 分配一个新块
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())           # 跨入新块 → 从 free 队头取一块；其余情况什么都不做

    def hash_blocks(self, seq: Sequence):
        # 每步 postprocess 中调用（prefill 和 decode 都会经过），此时本步的 KV 已经写入 cache
        # 本步计算的范围是 [num_cached_tokens, num_cached_tokens + num_scheduled_tokens)，
        # 把这一步中被"写满"的块登记到 hash_to_block_id，之后的请求才能命中它们
        start = seq.num_cached_tokens // self.block_size             # ① 本步开始时所在的块（之前可能只写了一部分，还没登记）
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
                                                                     # ② 向下取整：只算写满的块，未写满的最后一块不登记
        if start == end: return                                      #    本步没有写满任何块（绝大多数 decode 步）
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
                                                                     # ③ 链式哈希的起点：前一块早已写满并登记过
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)                               # ④ 块记下自己的身份（_allocate_block 淘汰时要用）
            self.hash_to_block_id[h] = block.block_id                # ⑤ 登记全局索引；同前缀已登记过时会覆盖为这个块
