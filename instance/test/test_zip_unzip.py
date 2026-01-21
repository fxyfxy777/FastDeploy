import paddle
import fastdeploy
import paddle.nn.functional as F
import numpy as np
# ===================== 工具函数 =====================
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

# ===================== 配置参数 =====================
paddle.seed(10)

NUM_ROWS = 128
LOCAL_NUM_EXPERTS = 8
PADDING_ALIGNMENT = 128
HIDDEN_SIZE = 4096
SCALE_SIZE = HIDDEN_SIZE // 128

# ===================== 构造输入 =====================
x = paddle.randn(
    [NUM_ROWS, HIDDEN_SIZE], dtype="bfloat16"
).cast(paddle.float8_e4m3fn)

x_scale = paddle.randn(
    [NUM_ROWS, SCALE_SIZE], dtype="float32"
)

gate_out = paddle.randn(
    [NUM_ROWS, LOCAL_NUM_EXPERTS], dtype="float32"
)

topk_weights, topk_idx = paddle.topk(
    gate_out, k=8, axis=-1
)

topk_idx[:, 3:5] = -1

# ===================== tokens_per_expert =====================
token_count_per_expert = [0] * LOCAL_NUM_EXPERTS
for idx in topk_idx.flatten().numpy().tolist():
    if idx >= 0:
        token_count_per_expert[idx] += 1

token_count_per_expert_padded = [
    (c + PADDING_ALIGNMENT - 1) // PADDING_ALIGNMENT * PADDING_ALIGNMENT
    for c in token_count_per_expert
]

token_all_num = sum(token_count_per_expert_padded)

token_count_per_expert_tensor = paddle.to_tensor(
    token_count_per_expert, dtype="int32"
)
token_count_per_expert_padded_tensor = paddle.to_tensor(
    token_count_per_expert_padded, dtype="int32"
)

# ===================== FastDeploy 路径 =====================
def fd_fun():
    (
        permute_input,
        permute_scale,
        permute_indices_per_token,
        num_tokens_per_expert_list_cumsum,
        num_tokens_per_expert_list_padded_cumsum,
        token_prob_unzipped,          # == dst_weights
        dst_indices,
        cumsum_idx_gpu,
        m_indices,
    ) = fastdeploy.model_executor.ops.gpu.ep_moe_expert_dispatch_fp8(
        x,
        x_scale,
        topk_idx,
        topk_weights,
        token_count_per_expert_tensor,
        token_count_per_expert_padded_tensor,
        True,
        token_all_num,
    )

    expert_out = permute_input.astype("bfloat16")

    final_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
        expert_out,
        token_prob_unzipped,
        permute_indices_per_token,
        dst_indices,
        None,
        False,
        1.0,
    )

    dump_tensor(        x,        x_scale,        topk_idx,        topk_weights,        permute_input,        permute_scale,        permute_indices_per_token,        token_prob_unzipped,        dst_indices,        m_indices,        final_out,            )

    return final_out

# ===================== Paddle 路径 =====================
def pd_fun():
    (
        permute_input,
        permute_indices_per_token,     # == zipped_expertwise_rowmap
        token_prob_unzipped,
        permute_scale,
    ) = F.moe_permute(
        hidden_states=x,
        scale=x_scale,
        expert_routemap_topk=topk_idx.astype("int32"),
        expert_prob_topk=topk_weights,
        num_experts=LOCAL_NUM_EXPERTS,
        tokens_per_expert=token_count_per_expert_padded,
        padding_alignment=PADDING_ALIGNMENT,
        do_gather=True,
    )

    expert_out = permute_input.astype("bfloat16")
    hidden_states_unzipped = (expert_out.astype("float32") * token_prob_unzipped.astype("float32").unsqueeze(-1)).astype("bfloat16")
    out_tokens, out_probs = F.moe_unpermute(
        hidden_states_unzipped=hidden_states_unzipped,
        zipped_expertwise_rowmap=permute_indices_per_token,
        expert_routemap_topk=topk_idx.astype("int32"),
        token_prob_unzipped=token_prob_unzipped,
        total_zipped_tokens=NUM_ROWS,
        num_experts=LOCAL_NUM_EXPERTS,
        use_mix_precision = True,
    )

    dump_tensor(        x,        x_scale,        topk_idx,        topk_weights,        permute_input,        permute_scale,        permute_indices_per_token,        token_prob_unzipped,        expert_out,        out_tokens,        out_probs    )

    return out_tokens

# ===================== 执行 =====================
fd_out = fd_fun()
pd_out = pd_fun()
print(100 * "*")
# print(fd_out)
# print(pd_out)
np.testing.assert_allclose(fd_out.numpy(), pd_out.numpy(), rtol=1e-05, atol=1e-06)