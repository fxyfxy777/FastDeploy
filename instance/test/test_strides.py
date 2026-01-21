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

    

x = paddle.randn([5, 7])

y = x.T

z = y[:4]

dump_tensor(x, y, z)