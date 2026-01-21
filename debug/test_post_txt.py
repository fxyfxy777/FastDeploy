import openai

ip = "0.0.0.0"
service_http_port = "8188"    # 服务配置的
client = openai.Client(base_url=f"http://{ip}:{service_http_port}/v1", api_key="EMPTY_API_KEY")

# 流式对话，历史多轮
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "你好，请问你是谁？"
        },
    ],
    temperature=1,
    max_tokens=32000,
    stream=True,
    metadata={"enable_thinking": False},
)

def get_str(content_raw):
    content_str = str(content_raw) if content_raw is not None else ''
    return content_str

cnt = 0
for chunk in response:
    # print(chunk)
    if chunk.choices[0].delta is not None and chunk.choices[0].delta.role != 'assistant':
        reasoning_content = get_str(chunk.choices[0].delta.reasoning_content)
        content = get_str(chunk.choices[0].delta.content)
        print(reasoning_content+content, end='', flush=True)
        if chunk.choices[0].delta.token_ids[0] == 100282:
            print("\nAnswer:")
        cnt += 1
print(f"\ntotal tokens: {cnt}")