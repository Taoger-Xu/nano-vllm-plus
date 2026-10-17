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
        """初始化一个 rank 的模型执行器：加入进程组、加载模型、warmup、分配 KV cache、录制 CUDA Graph。

        rank > 0 的子进程在构造函数末尾进入 loop()，直到收到 exit 才返回。

        Args:
            config (Config): 引擎配置；子进程拿到的是 rank 0 的一份 pickle 拷贝。
            rank (int): 本进程的 TP rank，同时也是使用的 GPU 编号。
            event (Event | list[Event]): rank 0 持有所有子进程的 Event 列表，用于通知；子进程只持有自己的那一个。
        """
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
        """释放共享内存、CUDA Graph 和进程组。由 LLMEngine.exit 通过 call("exit") 在所有 rank 上执行。"""
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
        """rank > 0 的主循环：等待 rank 0 的指令 → 执行同名方法 → 再等待，收到 exit 后退出。"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)                            # rank > 0 时 call 不会再写共享内存，只在本地执行；返回值丢弃
            if method_name == "exit":
                break

    def read_shm(self):
        """rank > 0：阻塞等待 rank 0 的通知，从共享内存中读出一条指令。

        没有"已读确认"：rank 0 写下一条消息之前，要靠本次方法中的集合通信（run 的前向、exit 的 barrier）
        保证子进程已经读完并 clear。所以每个通过 call 调用的方法都必须至少包含一次集合通信。

        Returns:
            tuple[str, list]: 方法名和参数列表；seqs 按 Sequence.__getstate__ 精简反序列化。
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()                                            # 阻塞，直到 rank 0 set
        n = int.from_bytes(self.shm.buf[0:4], "little")              # 消息格式：[0, 4) 长度 n，[4, 4+n) pickle 数据
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()                                           # 复位，准备等下一条
        return method_name, args

    def write_shm(self, method_name, *args):
        """rank 0：把一条指令写进共享内存，并通知所有子进程。

        消息格式：[0, 4) 为长度 n，[4, 4+n) 为 pickle.dumps([method_name, *args])。

        Args:
            method_name (str): 要在所有 rank 上执行的方法名。
            *args: 方法的参数。
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()                                              # 通知每一个子进程

    def call(self, method_name, *args):
        """SPMD 式的广播调用：rank 0 先把指令写进共享内存，然后本 rank 执行同一个方法。

        目前只有 run 和 exit 通过它调用。

        Args:
            method_name (str): 方法名。
            *args: 方法的参数。

        Returns:
            Any: 方法的返回值；只在 rank 0 有意义（run 返回 token_ids，其他 rank 返回 None）。
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    @torch.inference_mode()
    def warmup_model(self):
        """用假数据跑一次最大规模的 prefill（总 token 数 = max_num_batched_tokens），测出激活显存峰值。

        allocate_kv_cache 会把这个峰值预留出来，剩下的显存才分给 KV cache。
        开启草稿模型时两个模型都执行一遍，草稿模型的激活也计入峰值。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)                 # 单条序列不能超过 max_model_len
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)   # 条数同时受两个上限约束
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        input_ids, positions = self.prepare_prefill(seqs)
        self.model(input_ids, positions)
        if self.draft_model is not None:
            self.draft_model(input_ids, positions)
        reset_context()
        torch.cuda.empty_cache()

    def get_kv_cache_block_bytes(self, hf_config):
        """计算一个逻辑 KV cache block 所需的显存字节数。

        Args:
            hf_config (AutoConfig): 当前模型的 Hugging Face 配置。

        Returns:
            int: 当前模型每个 KV cache block 的字节数。
        """
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        return 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize

    def bind_kv_cache(self, model, kv_cache, hf_config):
        """将模型中每一层 Attention 绑定到对应的 KV cache 切片。

        Args:
            model (torch.nn.Module): 要绑定 KV cache 的 Transformer 模型。
            kv_cache (torch.Tensor): 形状为 ``[2, layers, blocks, ...]`` 的 KV cache。
            hf_config (AutoConfig): 当前模型的 Hugging Face 配置，用于校验层数。

        Returns:
            None: 该函数直接修改模型中 Attention 模块的 cache 引用。
        """
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == hf_config.num_hidden_layers

    def allocate_kv_cache(self):
        """用 warmup 之后剩余的显存分配 KV cache，并绑定到每一层 Attention。

        KV cache 是一个大张量：[2（K、V）, num_layers, num_blocks, block_size, num_kv_heads // tp_size, head_dim]，
        每层 Attention 通过 kv_cache[0, layer_id] / kv_cache[1, layer_id] 拿到自己的那一片。
        开启草稿模型时两个模型各自拥有一块物理 KV cache，由两个 BlockManager 分别管理；
        显存按两个模型每块字节数之和划分，两个池块数相同。
        结果写回 config.num_kvcache_blocks / num_draft_kvcache_blocks，供 Scheduler 使用。
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        block_bytes = self.get_kv_cache_block_bytes(hf_config)
        if self.draft_model is not None:
            block_bytes += self.get_kv_cache_block_bytes(self.speculative_config.hf_config)
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        self.bind_kv_cache(self.model, self.kv_cache, hf_config)

        if self.draft_model is not None:
            config.num_draft_kvcache_blocks = config.num_kvcache_blocks
            draft_config = self.speculative_config.hf_config
            draft_num_kv_heads = draft_config.num_key_value_heads // self.world_size
            draft_head_dim = getattr(draft_config, "head_dim", draft_config.hidden_size // draft_config.num_attention_heads)
            self.draft_kv_cache = torch.empty(2, draft_config.num_hidden_layers, config.num_draft_kvcache_blocks, self.block_size, draft_num_kv_heads, draft_head_dim)
            self.bind_kv_cache(self.draft_model, self.draft_kv_cache, draft_config)

    def prepare_block_tables(self, seqs: list[Sequence], role: str = "target"):
        """把各序列的 block_table 按本批最长的那条补 -1，拼成二维张量。

        flash-attn 需要规整的 [序列数, 最大块数] 张量；补的 -1 不会被读到，
        kernel 对每条序列只读它实际长度所需的那几个条目。

        Args:
            seqs (list[Sequence]): 本步的序列。
            role (str): 取哪个模型的 block_table，"target" 或 "draft"。

        Returns:
            torch.Tensor: CUDA 上的 int32 张量，形状为 ``[len(seqs), max_num_blocks]``。
        """
        tables = [seq.kv(role).block_table for seq in seqs]
        max_len = max(len(table) for table in tables)
        block_tables = [table + [-1] * (max_len - len(table)) for table in tables]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence], role: str = "target"):
        """把seqs prefill 的输入，即GPU中前向计算所需的 input_ids、positions 等张量。

        每个seq本步计算 [num_cached_tokens, num_cached_tokens + num_scheduled_tokens)
        只负责这部分新 token，K/V 覆盖 [0, end)；
        有前缀（prefix cache 命中或 chunked prefill 的后续块）时
        生成 block_tables，让 attention 从分页 cache 中读取 K/V。

        Args:
            seqs (list[Sequence]): 本步要 prefill 的序列。
            role (str): 按哪个模型的 KV 状态（进度、block_table）准备，"target" 或 "draft"。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 拼接成一维的 input_ids 和 positions，均在 CUDA 上。
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            kv = seq.kv(role)
            start = kv.num_cached_tokens
            seqlen_q = kv.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # warmup 时的假序列没有分配块（KV cache 也还没分配），跳过，slot_mapping 为空
            if not kv.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            # 循环未命中的逻辑块
            for i in range(start_block, end_block):
                slot_start = kv.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = kv.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = kv.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs, role)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """构造 decode 的输入（每条序列 1 个 token），并把 slot_mapping、context_lens、block_tables 写入 Context。

        只用于大模型：草稿模型的 decode 输入由推测路径单独准备。

        Args:
            seqs (list[Sequence]): 本步要 decode 的序列。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 每条序列的 last_token 和它的位置 len(seq) - 1，均在 CUDA 上。
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            pos = len(seq) - 1
            input_ids.append(seq.last_token)
            positions.append(pos)
            context_lens.append(len(seq))
            # 按位置查块，而不是取 block_table[-1]：推测步之后 block_table 末尾可能有多留的块
            slot_mapping.append(seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """收集每条序列的采样温度。

        Args:
            seqs (list[Sequence]): 本步的序列。

        Returns:
            torch.Tensor: CUDA 上的 float32 张量，形状为 ``[len(seqs)]``。
        """
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """执行大模型前向并计算 logits。

        prefill、enforce_eager 或 batch > 512 时走 eager；否则把数据拷进 graph_vars，replay 对应档位的 CUDA Graph。

        Args:
            input_ids (torch.Tensor): CUDA 上的输入 token。
            positions (torch.Tensor): CUDA 上的位置索引。
            is_prefill (bool): 是否为 prefill。

        Returns:
            torch.Tensor: logits；TP > 1 时只有 rank 0 拿到完整结果，其他 rank 为 None。
        """
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

    @torch.inference_mode()
    def run_prefill_models(self, seqs: list[Sequence], input_ids: torch.Tensor, positions: torch.Tensor):
        """执行 prefill：大模型前向并计算 logits；开启草稿模型时，草稿模型按自己的块表再前向一次，只写 KV cache。

        两个模型的块表不同，草稿模型必须用 prepare_prefill(seqs, "draft") 重新生成 slot_mapping / block_tables，
        不能复用大模型的 Context。

        Args:
            seqs (list[Sequence]): 本步要 prefill 的序列。
            input_ids (torch.Tensor): 大模型的输入 token（由 prepare_prefill(seqs) 生成，Context 已就绪）。
            positions (torch.Tensor): 大模型的位置索引。

        Returns:
            torch.Tensor: 大模型的 logits；TP > 1 时只有 rank 0 拿到完整结果。
        """
        # 先用大模型的 Context 算完 logits（LMHead 要读 cu_seqlens_q），再切换到草稿模型的 Context
        logits = self.model.compute_logits(self.model(input_ids, positions))
        if self.draft_model is not None:
            reset_context()
            draft_input_ids, draft_positions = self.prepare_prefill(seqs, role="draft")
            self.draft_model(draft_input_ids, draft_positions)
        return logits

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """执行一步：构造输入 → 前向（prefill 时草稿模型一起前向）→ rank 0 采样。通过 call 在所有 rank 上执行。

        Args:
            seqs (list[Sequence]): Scheduler 本步调度出的序列。
            is_prefill (bool): 本步是 prefill 还是 decode。

        Returns:
            list[int]: 每条序列采样出的 token；只在 rank 0 有值，其他 rank 返回 None。
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        if is_prefill:
            logits = self.run_prefill_models(seqs, input_ids, positions)
        else:
            logits = self.run_model(input_ids, positions, False)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """启动时为 decode 的各档 batch size 录制 CUDA Graph，所有档位共用一个内存池，从大到小录制。

        录制时用全 0 的假数据，Context 中放的是 graph_vars 的切片；之后每步只需把数据拷进 graph_vars 再 replay。
        """
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
