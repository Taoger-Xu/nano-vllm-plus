import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0                              # 头数、中间维度等必须能被 TP 整除
    return numerator // denominator


class LinearBase(nn.Module):
    """所有并行线性层的基类，计算 Y = X · Wᵀ，W 的形状是 [out, in]。

    它做了三件事：
    1. 记录本 rank 的编号 tp_rank、总数 tp_size，以及沿哪个维度切 tp_dim；
    2. 只按本 rank 那一片的大小分配权重（子类传进来的 input_size / output_size 已经是切分后的大小）；
    3. 把子类的 weight_loader 挂在参数上，loader.py 加载时调用它，从完整的 checkpoint 中切出本 rank 那一片。

    所有子类的 weight_loader(param, loaded_weight[, shard_id]) 参数含义相同：
        param：        目标。模型中已分配好的参数（weight 或 bias，两者挂的是同一个 loader），形状是本 rank 那一片，在 GPU 上；
        loaded_weight：来源。从 checkpoint 读出的权重，形状是完整的，在 CPU 上；
        shard_id：     只有合并的层（Merged / QKV）才有，指明这次加载的是哪个子矩阵、该放进 param 的哪个位置。
    做的都是同一件事：从 loaded_weight 中取出本 rank 需要的部分，用 param.data.copy_ 原地拷贝进 param；
    各子类的区别只在于"取哪部分、放到哪里"。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim                                         # 沿哪个维度切：0 = 输出（列并行），1 = 输入（行并行），None = 不切
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))   # 子类传进来的已经是切分后的大小
        # 给参数对象加一个名为 weight_loader 的属性
        self.weight.weight_loader = self.weight_loader               # 挂在参数上，loader.py 加载时调用
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """不切：每个 rank 存完整权重，前向无通信。

    适合不值得切的小矩阵（如 MoE 的路由层），Qwen3 中未使用。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """列并行：W 沿输出维度（dim 0）切，rank i 存 W_i [out/N, in]。

    输入 X：每个 rank 都是完整的；
    输出 Y_i = X · W_iᵀ：[tokens, out/N]，只有输出特征的第 i 段，前向无通信。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)                    # out/N
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)   # 完整权重的第 tp_rank 段
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 不做 all-gather：后面总是接行并行层，它需要的输入恰好是切开的 Y_i
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """多个列并行矩阵拼成一个（gate_proj + up_proj → gate_up_proj），一次 GEMM 算完。

    不能把拼接后的大矩阵整体切 N 段：TP=2 时 rank 0 会拿到全部 gate、rank 1 拿到全部 up，无法计算 SiLU(gate) × up。
    而要每个子矩阵分别切，再把各自的第 i 段拼到 rank i：
        rank 0 = [gate 第 1 个 1/N | up 第 1 个 1/N]
        rank 1 = [gate 第 2 个 1/N | up 第 2 个 1/N]
        ...
    这样每个 rank 上的 gate 和 up 对应同一段中间维度，可以直接逐元素相乘。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes                             # 各子矩阵的完整输出大小，如 [inter, inter]
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # 每个子矩阵各调用一次；loaded_shard_id 由 Qwen3ForCausalLM.packed_modules_mapping 给出（gate=0，up=1）
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size   # 这个子矩阵在本 rank 参数中的起始行
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size           # 它在本 rank 上占多少行
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]   # 子矩阵自己切 N 份，取第 rank 份
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """q_proj、k_proj、v_proj 拼成一个列并行矩阵，并且按"头"切。

    每个 rank 拿到完整的若干个头，attention 才能按头独立计算；GQA 中共享同一个 kv 头的 q 头会落在同一个 rank 上。
    例：Qwen3-8B（32 个 q 头、8 个 kv 头、head_dim 128），TP=2 时每个 rank 的权重为
        [ q：16 头 × 128 = 2048 行 | k：4 头 × 128 = 512 行 | v：4 头 × 128 = 512 行 ]
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)            # 本 rank 的 q 头数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)      # 本 rank 的 kv 头数：TP 不能超过 kv 头数
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        # 与 MergedColumnParallelLinear 相同的"分别切、再拼接"，只是 q、k、v 三段大小不同
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]   # 按头的边界切，取第 rank 份
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """行并行：W 沿输入维度（dim 1）切，rank i 存 W_i [out, in/N]。

    输入 X_i：前一个列并行层输出的第 i 段 [tokens, in/N]；
    输出 Y_i = X_i · W_iᵀ：[tokens, out]，形状完整但只是部分和，Y = Σ Y_i 需要 all-reduce。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:                                     # bias 的长度是 out，不切，每个 rank 都加载完整的一份
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)                    # in/N
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bias 只由 rank 0 加：all-reduce 是求和，每个 rank 都加的话 bias 会被加 N 次
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)   # 部分和 Y_i
        if self.tp_size > 1:
            dist.all_reduce(y)                                       # Y = Σ Y_i，每个 rank 都拿到完整结果，也就是下一层的完整输入
        return y
