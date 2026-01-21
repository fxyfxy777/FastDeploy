


# export LD_LIBRARY_PATH=/usr/local/nccl:$LD_LIBRARY_PATH
# export CUDA_VISIBLE_DEVICES=0
# export FD_USE_DEEP_GEMM=1
# python -m fastdeploy.entrypoints.openai.api_server \
#     --model /workspace3/chenjianye/models/ERNIE-4.5-21B-A3B-Paddle/ \
#     --tensor-parallel-size 1 \
#     --max-model-len 32768 \
#     --max-num-seqs 128 \
#     --load-choices "default_v1" \
#     --graph-optimization-config '{"use_cudagraph":false}' \
#     --port 8188 \
#     --quantization "block_wise_fp8" \
#     --num-gpu-blocks-override 5000


# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export FD_USE_DEEP_GEMM=1
# python -m fastdeploy.entrypoints.openai.api_server \
#     --model ~/PaddlePaddle/ERNIE-4.5-300B-A47B-Paddle \
#     --tensor-parallel-size 1\
#     --max-model-len 32768 \
#     --max-num-seqs 96 \
#     --load-choices "default" \
#     --graph-optimization-config '{"use_cudagraph":true}' \
#     --port 8188 \
#     --quantization "block_wise_fp8" \
#     --enable-expert-parallel \
#     --data-parallel-size 8  \
#     --engine-worker-queue-port "6077,6078,6079,6080,6081,6082,6083,6084" \
#     --disable-custom-all-reduce
export PYTHONPATH=$PWD:$PYTHONPATH
rm -rf log/*
rm core.* -f
export CUDA_VISIBLE_DEVICES=5
export FD_USE_DEEP_GEMM=1
# /workspace3/chenjianye/nsight_2025_5_1/bin/nsys launch \
#   --cuda-graph-trace=node \
#   --session=fxy \
python -m fastdeploy.entrypoints.openai.api_server \
  --model PaddlePaddle/ERNIE-4.5-21B-A3B-Paddle \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --load-choices "default_v1" \
  --graph-optimization-config '{"use_cudagraph":true}' \
  --port 8286 \
  --quantization "block_wise_fp8" \


