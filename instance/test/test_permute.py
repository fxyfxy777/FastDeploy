import paddle
import numpy as np
import paddle.nn.functional as F
import fastdeploy

# ===================== 工具函数（独立封装）=====================
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

# ===================== 全局配置（统一管理，便于修改）=====================
paddle.seed(10)  # 固定随机种子，保证结果可复现
NUM_ROWS = 128
LOCAL_NUM_EXPERTS = 8
PADDING_ALIGNMENT = 128
HIDDEN_SIZE = 4096
SCALE_SIZE = HIDDEN_SIZE // 128

# ===================== 1. 生成测试数据（两个算子共用的输入，一次性生成）=====================
print("===== 生成测试输入数据 =====")
# 核心输入张量
recv_x = paddle.randn([NUM_ROWS, HIDDEN_SIZE], dtype="bfloat16").cast(paddle.float8_e4m3fn)
recv_x_scale = paddle.randn([NUM_ROWS, SCALE_SIZE]).cast("float32")
gate_out = paddle.randn([NUM_ROWS, LOCAL_NUM_EXPERTS], dtype="float32")

# TopK相关计算（索引和权重）
recv_topk_idx = paddle.topk(gate_out, k=8, axis=-1)[1]
recv_topk_idx[:, 3:5] = -1  # 手动设置部分索引为-1，模拟无效专家索引
recv_topk_weights = paddle.topk(gate_out, k=8, axis=-1)[0]

# 计算每个专家的token数及对齐后的数量（padding到128的倍数）
tmp0 = [0] * LOCAL_NUM_EXPERTS
recv_topk_idx_list = recv_topk_idx.flatten().numpy().tolist()
for ele in recv_topk_idx_list:
    if ele >= 0:
        tmp0[ele] += 1

tmp1 = [(cnt + PADDING_ALIGNMENT - 1) // PADDING_ALIGNMENT * PADDING_ALIGNMENT for cnt in tmp0]
token_all_num = sum(tmp1)

# 转换为paddle张量
tmp0_tensor = paddle.to_tensor(tmp0).cast("int32")
tmp1_tensor = paddle.to_tensor(tmp1).cast("int32")
tokens_per_expert = tmp0  # 非张量版本，供baseline算子使用

# 打印基础信息
print(f"每个专家的原始token数: {tmp0}")
print(f"每个专家对齐后的token数: {tmp1}")
print(f"所有专家的总token数: {token_all_num}")
dump_tensor(recv_x, recv_x_scale, recv_topk_idx, recv_topk_weights, tmp0_tensor, tmp1_tensor)

# ===================== 2. 基准算子：F.moe_permute 计算（隔离模块）=====================
print("\n===== 执行基准算子 F.moe_permute =====")
(baseline_hidden_states_unzipped,
baseline_zipped_expertwise_rowmap,
baseline_token_prob_unzipped,
baseline_scale_unzipped )= F.moe_permute(
    hidden_states=recv_x,
    scale=recv_x_scale,
    expert_routemap_topk=recv_topk_idx.astype(paddle.int32),
    expert_prob_topk=recv_topk_weights,
    num_experts=LOCAL_NUM_EXPERTS,
    tokens_per_expert=tokens_per_expert,
    padding_alignment=PADDING_ALIGNMENT,
    do_gather=True,
)
# 打印基准算子输出
dump_tensor(baseline_hidden_states_unzipped,baseline_zipped_expertwise_rowmap,baseline_token_prob_unzipped,baseline_scale_unzipped)

# ===================== 3. 测试算子：fastdeploy ep_moe_expert_dispatch_fp8 计算（隔离模块）=====================
print("\n===== 执行测试算子 fastdeploy.ep_moe_expert_dispatch_fp8 =====")
(test_permute_input,
test_permute_scale,
test_permute_indices_per_token,
test_recv_num_tokens_per_expert_list_cumsum,
test_recv_num_tokens_per_expert_list_padded_cumsum,
test_dst_weights,
test_dst_indices,
test_cumsum_idx_gpu,
test_m_indices) = fastdeploy.model_executor.ops.gpu.ep_moe_expert_dispatch_fp8(
    recv_x,
    recv_x_scale,
    recv_topk_idx,
    recv_topk_weights,
    tmp0_tensor,
    tmp1_tensor,
    True,  # use_in_ep
    token_all_num,
)

fd_output = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
    ffn_out, dst_weights, permute_indices_per_token, dst_indices,
    None, False, 1.0
)
# 打印测试算子输出
dump_tensor(test_permute_input,test_permute_scale,test_permute_indices_per_token,test_recv_num_tokens_per_expert_list_cumsum,test_recv_num_tokens_per_expert_list_padded_cumsum,test_dst_weights,test_dst_indices,test_cumsum_idx_gpu,test_m_indices)


def build_m_indices_from_paddle(
    baseline_zipped_expertwise_rowmap,
    permute_rows  # = baseline_hidden_states_unzipped.shape[0]
):
    """
    从 Paddle 的 moe_permute 输出构造 m_indices
    """
    num_rows, num_experts = baseline_zipped_expertwise_rowmap.shape
    # 初始化为 -1（padding）
    m_indices = paddle.full(
        [permute_rows],
        -1,
        dtype=baseline_zipped_expertwise_rowmap.dtype,
    )
    # rowmap[t, e] = r  ==>  m_indices[r] = e
    for e in range(num_experts):
        rows = baseline_zipped_expertwise_rowmap[:, e]
        valid_mask = rows >= 0
        valid_rows = rows[valid_mask]
        m_indices[valid_rows] = e
    return m_indices

baseline_m_indices = build_m_indices_from_paddle(
    baseline_zipped_expertwise_rowmap,
    baseline_hidden_states_unzipped.shape[0]
)
# print(test_m_indices[128:128*2])
# print(baseline_m_indices[128:128*2])
np.testing.assert_allclose(
        baseline_m_indices.numpy(),
        test_m_indices.numpy(),
        rtol=1e-3,
        atol=1e-5,
        err_msg="test_m_indices 精度对齐失败"
    )
# ===================== 4. 精度对齐校验（隔离模块，清晰标注校验项）=====================
print(baseline_zipped_expertwise_rowmap)
# print(test_dst_indices)
print(test_permute_indices_per_token.T)


print("\n===== 开始精度对齐校验 =====")
try:
    num_experts = LOCAL_NUM_EXPERTS
    num_tokens = recv_x.shape[0]

    for e in range(num_experts):
        for t in range(num_tokens):
            paddle_uz = int(baseline_zipped_expertwise_rowmap.T[e, t])
            fd_uz = int(test_permute_indices_per_token[e, t])

            # Paddle 认为 (e, t) 是有效的
            if paddle_uz >= 0:
                assert fd_uz >= 0, (
                    f"Token {t} in expert {e} exists in Paddle "
                    f"but missing in FastDeploy"
                )
            else:
                # Paddle 认为不属于这个 expert
                assert fd_uz < 0, (
                    f"Token {t} in expert {e} does NOT exist in Paddle "
                    f"but appears in FastDeploy"
                )


except AssertionError as e:
    print(f"❌ 精度对齐失败: {e}")
except Exception as e:
    print(f"❌ 校验过程出错: {e}")