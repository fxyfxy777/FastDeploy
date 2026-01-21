config_yaml=benchmarks/yaml/eb45-32k-blockwise-fp8-h800-tp8.yaml
model_path=debug/ERNIE-4.5-300B-A47B-Paddle
python -m fastdeploy.entrypoints.openai.api_server --model ${model_path} --port 8188 --metrics-port 8005 --cache-queue-port 55663 --config ${config_yaml}