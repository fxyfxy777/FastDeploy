import paddle
import paddle.nn.functional as F
import numpy as np

# ===================== 配置参数 =====================
paddle.seed(10)

NUM_ROWS = 128
LOCAL_NUM_EXPERTS = 8
TOPK = 8
PADDING_ALIGNMENT = 128

HIDDEN_SIZE = 4096
SCALE_SIZE = HIDDEN_SIZE // 128

# ===================== 1. 构造输入 =====================
print("===== Generate Inputs =====")

# hidden states (fp8)
recv_x = paddle.randn(
    [NUM_ROWS, HIDDEN_SIZE], dtype="bfloat16"
).cast(paddle.float8_e4m3fn)

# fp8 scale
recv_x_scale = paddle.randn(
    [NUM_ROWS, SCALE_SIZE], dtype="float32"
)

# gate output
gate_out = paddle.randn(
    [NUM_ROWS, LOCAL_NUM_EXPERTS], dtype="float32"
)

# topk
recv_topk_weights, recv_topk_idx = paddle.topk(
    gate_out, k=TOPK, axis=-1
)

# 手动制造无效 expert
recv_topk_idx[:, 3:5] = -1
recv_topk_weights = paddle.where(
    recv_topk_idx >= 0,
    recv_topk_weights,
    paddle.zeros_like(recv_topk_weights),
)

# ===================== 2. tokens_per_expert =====================
tmp0 = [0] * LOCAL_NUM_EXPERTS
for idx in recv_topk_idx.flatten().numpy().tolist():
    if idx >= 0:
        tmp0[idx] += 1

tokens_per_expert = tmp0

print("tokens_per_expert:", tokens_per_expert)

# ===================== 3. Paddle moe_permute（FD dispatch 对齐） =====================
print("\n===== Paddle moe_permute =====")

(
    permute_input,
    zipped_expertwise_rowmap,
    token_prob_unzipped,
    permute_scale,
) = F.moe_permute(
    hidden_states=recv_x,
    scale=recv_x_scale,
    expert_routemap_topk=recv_topk_idx.astype("int32"),
    expert_prob_topk=recv_topk_weights,
    num_experts=LOCAL_NUM_EXPERTS,
    tokens_per_expert=tokens_per_expert,
    padding_alignment=PADDING_ALIGNMENT,
    do_gather=True,
)

print("permute_input shape:", permute_input.shape)
print("zipped_expertwise_rowmap shape:", zipped_expertwise_rowmap.shape)

# ===================== 4. Expert 内计算（示例：直通） =====================
# 实际中这里是 FFN / Linear / GEMM
ffn_out = permute_input.astype("bfloat16")

# ===================== 5. Paddle moe_unpermute（FD combine 前半段） =====================
print("\n===== Paddle moe_unpermute =====")

out_tokens, out_probs = F.moe_unpermute(
    ffn_out,
    zipped_expertwise_rowmap,
    recv_topk_idx.astype("int32"),
    token_prob_unzipped,
    NUM_ROWS,
    LOCAL_NUM_EXPERTS,
)

print("out_tokens shape:", out_tokens.shape)
print("out_probs shape:", out_probs.shape)

# ===================== 6. 显式补 sum(input * weight)（FD combine 语义） =====================
print("\n===== Explicit Weighted Sum =====")

final_out = (
    out_tokens.astype("float32")
    * out_probs.unsqueeze(-1).astype("float32")
).sum(axis=1)

print("final_out shape:", final_out.shape)

# ===================== 7. 数值 sanity check =====================
print("\n===== Sanity Check =====")
print("final_out mean:", final_out.mean().item())
print("final_out std :", final_out.std().item())