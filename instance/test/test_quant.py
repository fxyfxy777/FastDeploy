import paddle
import fastdeploy

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

paddle.seed(100)
x_origin = paddle.randn((2, 128 * 5 ), dtype='bfloat16')

# x1, x_scale_tensor1= fastdeploy.model_executor.ops.gpu.per_token_quant_padding(
#     x_origin, 128, True
# )

# # x_scale_tensor1 = x_scale_tensor1[: x1.shape[0], ...]
# # print(x_scale_tensor1)
# # print('x1, ', x1)
# print('x_scale_tensor1', x_scale_tensor1)
# dump_tensor(x1, x_scale_tensor1,x_origin)


x2, x_scale_tensor2 = paddle.incubate.nn.functional.fp8_quant_blockwise(
    x_origin, output_scale_transpose=True, using_pow2_scale = False
)
# x_scale_tensor2 = x_scale_tensor2.reshape([x_scale_tensor2.shape[1],x_scale_tensor2.shape[0]])
# x_scale_tensor2 = x_scale_tensor2.T.contiguous().T
# x_scale_tensor2 = x_scale_tensor2[: x2.shape[0], ...]
# print('x2, ', x2)
# x_scale_tensor2 = x_scale_tensor2.T
print('x_scale_tensor2', x_scale_tensor2)
dump_tensor(x2, x_scale_tensor2)
x_scale_T = x_scale_tensor2.T[: x2.shape[0]]
dump_tensor(x_scale_T)
