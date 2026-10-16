import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 数据面：所有 rank 在这里汇合建立 NCCL 进程组（rank 0 最后才构造，子进程会在这里等它）
        # 之后 warmup、录 CUDA Graph 都有 all-reduce，各 rank 执行同一段 __init__，集合通信的顺序天然一致
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)                                  # rank i 使用 GPU i
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        # 主模型初始化后增加草稿模型初始化
        self.speculative_config = config.speculative_config
        self.draft_model = None
        if (
            self.speculative_config is not None
            and self.speculative_config.method == "draft_model"
        ):
            spec = self.speculative_config
            self.draft_model = Qwen3ForCausalLM(spec.hf_config)
            load_model(self.draft_model, spec.model)
        self.sampler = Sampler()

        # warmup 需要测量两个模型同时执行时的激活峰值
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 控制面：rank 0 创建 1MB 共享内存，用 barrier 保证子进程在它创建之后才去打开
        # 名字 "nanovllm" 是固定的，所以同一台机器上不能同时运行两个 TP > 1 的实例
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()                                          # 子进程进入循环，直到收到 exit，构造函数才返回

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()                                           # 等所有 rank 都 close 之后，
            if self.rank == 0:
                self.shm.unlink()                                    #   才由 rank 0 把共享内存从系统中删除
        if not self.enforce_eager:
            del self.graphs, self.graph_pool                         # 图中录有 NCCL 通信，要在销毁进程组之前删除
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        # rank > 0 的主循环：等待 rank 0 的指令 → 执行同名方法 → 再等待
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)                            # rank > 0 时 call 不会再写共享内存，只在本地执行；返回值丢弃
            if method_name == "exit":
                break

    def read_shm(self):
        # 没有"已读确认"：rank 0 写下一条消息之前，要靠本次方法中的集合通信（run 的前向、exit 的 barrier）
        # 保证子进程已经读完并 clear。所以每个通过 call 调用的方法都必须至少包含一次集合通信
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()                                            # 阻塞，直到 rank 0 set
        n = int.from_bytes(self.shm.buf[0:4], "little")              # 消息格式：[0, 4) 长度 n，[4, 4+n) pickle 数据
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])       # seqs 按 Sequence.__getstate__ 精简反序列化
        self.event.clear()                                           # 复位，准备等下一条
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()                                              # 通知每一个子进程

    def call(self, method_name, *args):
        # SPMD 式的广播调用：rank 0 先把 [方法名, 参数] 写进共享内存，然后所有 rank 执行同一个方法
        # 目前只有 run 和 exit 通过它调用；返回值只在 rank 0 有意义（run 返回 token_ids，其他 rank 返回 None）
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    @torch.inference_mode()
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 构造一次最大规模的 prefill（总 token 数 = max_num_batched_tokens），测出激活峰值；
        # allocate_kv_cache 会把这个峰值预留出来，剩下的显存才分给 KV cache
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)                 # 单条序列不能超过 max_model_len
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)   # 条数同时受两个上限约束
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        input_ids, positions = self.prepare_prefill(seqs)
        # warmup 同时执行两个模型，把草稿模型的激活显存计入峰值。
        self.model(input_ids, positions)
        if self.draft_model is not None:
            self.draft_model(input_ids, positions)
        reset_context()
        torch.cuda.empty_cache()

    """
    真实的分配kv cache对应的张量
    实际的 KV Cache 是一个大张量，其物理布局
    """
    def get_kv_cache_block_bytes(self, hf_config):
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        return 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

    def bind_kv_cache(self, model, kv_cache, hf_config):
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == hf_config.num_hidden_layers

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 两个模型共用逻辑 block_table，但各自拥有独立的物理 KV cache。
        block_bytes = self.get_kv_cache_block_bytes(hf_config)
        if self.draft_model is not None:
            block_bytes += self.get_kv_cache_block_bytes(self.speculative_config.hf_config)
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        """
            kv_cache = torch.empty(
                2,                          # K 和 V
                num_layers,                 # 层数
                num_blocks,                 # 总块数
                block_size,                 # 每块 token 数
                num_kv_heads // tp_size,    # KV head 数（考虑张量并行）
                head_dim                    # head 维度
            )
            不同的attention 通过索引获取自己的kv cache module.k_cache = kv_cache[0, layer_id]
        """
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        self.bind_kv_cache(self.model, self.kv_cache, hf_config)

        if self.draft_model is not None:
            draft_config = self.speculative_config.hf_config
            draft_num_kv_heads = draft_config.num_key_value_heads // self.world_size
            draft_head_dim = getattr(draft_config, "head_dim", draft_config.hidden_size // draft_config.num_attention_heads)
            self.draft_kv_cache = torch.empty(2, draft_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, draft_num_kv_heads, draft_head_dim)
            self.bind_kv_cache(self.draft_model, self.draft_kv_cache, draft_config)
    """
    因为要传给 flash-attn 的是一个规整的二维张量，
    而每条序列的块数不同，长短不一的列表拼不成矩阵
    """
    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # warmup 时的假序列没有分配块（KV cache 也还没分配），跳过，slot_mapping 为空
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            # 循环未命中的逻辑块
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run_prefill_models(self, input_ids: torch.Tensor, positions: torch.Tensor):
        # 两个模型复用同一套输入和 Context；Attention 的 cache 已分别绑定。
        target_hidden_states = self.model(input_ids, positions)
        if self.draft_model is not None:
            # 草稿模型此处只写 KV cache，不计算 logits。
            self.draft_model(input_ids, positions)
        return target_hidden_states

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        if is_prefill:
            hidden_states = self.run_prefill_models(input_ids, positions)
            logits = self.model.compute_logits(hidden_states)
        else:
            logits = self.run_model(input_ids, positions, False)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)                          # decode batch 不会超过 max_num_seqs；超过 512 时走 eager
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
