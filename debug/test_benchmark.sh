unset http_proxy
unset https_proxy
dataset=debug/0419_api9_yiyan_spv5_forqianfan_4872_fd
request_yaml=benchmarks/yaml/request_yaml/eb45-32k.yaml
python benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --model EB45T \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port 8188 \
  --dataset-name EBChat \
  --dataset-path ${dataset} \
  --hyperparameter-path ${request_yaml} \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 4872 \
  --max-concurrency 100 \
  --drop-ratio 0.2 \
  --save-result --debug > "infer_log.txt" 2>&1