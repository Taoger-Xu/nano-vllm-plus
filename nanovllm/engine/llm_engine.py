import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        # spawn 而不是 fork：fork 出的子进程不能使用 CUDA。子进程拿到的是 config 的一份 pickle 拷贝
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):              # ① 先启动 rank 1..N−1
            event = ctx.Event()                                      #    每个子进程一个 Event，rank 0 用它通知"有新消息"
            process = ctx.Process(target=ModelRunner, args=(config, i, event))   # 子进程的入口就是 ModelRunner 的构造函数，
            process.start()                                          #   构造函数最后进入 loop()，直到 exit 才返回
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)     # ② rank 0 在主进程中构造，持有所有子进程的 Event
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)                           # ③ 最后创建：需要 rank 0 的 allocate_kv_cache 写回的 num_kvcache_blocks
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")                               # 广播 exit：各 rank 关闭共享内存、销毁进程组
        del self.model_runner
        for p in self.ps:
            p.join()                                                 # 等待子进程的 loop 结束、进程退出

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """执行一步：调度 → 所有 rank 执行 run → 后处理。

        Returns:
            tuple[list, int]: 本步完成的 (seq_id, completion_token_ids) 列表；以及 token 数，
                prefill 为正（本步计算的 token 数），decode 为负（序列数）。
        """
        seqs, is_prefill = self.scheduler.schedule()
        # 推测步（每条序列算 1 + K 个 token）必须走 run_spec / postprocess_spec；
        # 在它们实现之前，误走普通路径会让 num_cached_tokens 多加 K，与 KV cache 的实际内容不一致
        if not is_prefill and seqs[0].num_scheduled_tokens > 1:
            raise NotImplementedError("run_spec / postprocess_spec 尚未实现")
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
