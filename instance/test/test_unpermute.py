import paddle
import numpy as np
import fastdeploy
from paddle.incubate.nn.functional import moe_combine

def dump_tensor(*args):
    """打印张量的关键信息（类型、形状、 dtype、步长），用于调试精度问题"""
    import inspect
    frame = inspect.currentframe().f_back
    try:
        call = inspect.getframeinfo(frame).code_context[0]
        names = call[call.find('(')+1:call.rfind(')')].split(',')
    except Exception:
        names = [f"arg{i}" for i in range(len(args))]
    
    print(100 * "*")
    for i, x in enumerate(args):
        name = names[i].strip() if i < len(names) else f"arg{i}"
        print(
            f"[{name:<60}] "
            f"type={type(x).__name__:<12} "
            f"shape={tuple(x.shape)!s:<18} "
            f"dtype={str(x.dtype):<20} "
            f"strides={x.strides}"
        )

# ===================== 1. 固定全局种子（保证数据唯一）=====================
SEED = 1234
paddle.seed(SEED)
np.random.seed(SEED)

# ===================== 2. 构造匹配维度的输入数据（核心修复：容量一致）=====================
# 核心维度（保证reshape容量匹配）
S = 8192          # 输出维度
K = 64            # topk维度
DIM = 2560        # 特征维度
SEQ = S * K       # 关键：让SEQ = S*K = 8192*64=524288，保证reshape容量匹配

# 构造基础输入（两个算子共用，维度容量匹配）
ffn_out = paddle.randn([SEQ, DIM], dtype=paddle.bfloat16)  # (524288, 2560)
dst_weights = paddle.randn([SEQ], dtype=paddle.float32)    # (524288,) → 可reshape为(8192,64)
permute_indices_per_token = paddle.randint(0, SEQ, [K, S], dtype=paddle.int32)  # (64, 8192)
dst_indices = paddle.randint(0, SEQ, [S, K], dtype=paddle.int32)               # (8192, 64)

# moe_combine专属适配（reshape容量匹配，无报错）
combine_weights = dst_weights.reshape([S, K]).cast("float32")  # (8192,64)，容量524288=SEQ
scatter_index = permute_indices_per_token                      # 直接复用(64,8192)


fd_output = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
    ffn_out, dst_weights, permute_indices_per_token, dst_indices,
    None, False, 1.0
)

dump_tensor(fd_output,ffn_out, dst_weights, permute_indices_per_token, dst_indices)

x_fp32 = ffn_out.cast("float32")

dump_tensor(x_fp32, combine_weights, scatter_index)
# ===================== 4. 调用paddle moe_combine算子 =====================

# ===== 前置权重计算（对齐 FastDeploy 语义）=====
x_reshaped = x_fp32.reshape([S, K, DIM])          # [8192,64,2560]
w = combine_weights.unsqueeze(-1)                 # [8192,64,1]
x_weighted = x_reshaped * w                       # [8192,64,2560]
x_weighted_flat = x_weighted.reshape([SEQ, DIM])  # [524288,2560]


pd_output = moe_combine(
    x=x_fp32,  # 兼容你原代码的float32
    combine_weights=combine_weights,
    scatter_index=scatter_index
)

# ===================== 5. 极简校验 =====================
assert fd_output.shape == (S, DIM), f"fastdeploy输出形状错误: {fd_output.shape}"
assert pd_output.shape == (S, DIM), f"paddle输出形状错误: {pd_output.shape}"
np.testing.assert_allclose(
    fd_output.numpy(),
    pd_output.numpy(),
    rtol=1e-3, atol=1e-3
)
print("✅ 无维度报错，两个算子共用同一数据，调用+校验成功！")