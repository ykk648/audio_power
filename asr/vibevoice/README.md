## VibeVoice

[vibevoice](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr.md)  微软 2026，  [hf](https://huggingface.co/microsoft/VibeVoice-ASR)

### demo

返回speaker ID, 返回时间戳

````commandline
✅ Transcription successful! Result saved to vibevoice_result.json

--- Result Preview ---
{'Start': 0, 'End': 7.17, 'Speaker': 0, 'Content': '行了，别演了。你到底是谁？图什么？要钱的话我真没有。'}
{'Start': 7.17, 'End': 16.59, 'Speaker': 1, 'Content': '我是您的专属仿生女，被投放到此，旨在协助您完成完美女友挑战。仿生人？你科幻片看多了吧，还挑战？你们这诈骗团伙剧本写得挺全啊。'}
{'Start': 16.59, 'End': 29.33, 'Content': '[Environmental Sounds]'}
...
````

### deploy

部署和测试参考 https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr.md

ubuntu22.04 cuda12.9

固定环境
```sh
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice

# 在当前目录新建文件 `Dockerfile`:
FROM vllm/vllm-openai:v0.14.1
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
ENV HF_ENDPOINT=https://hf-mirror.com

构建镜像：
docker build -t vibevoice-custom .
```

24G显存4090 4卡配置
```sh
# 修改 vllm_plugin/scripts/start_server.py

vllm_cmd = [
	"vllm", "serve", model_path,
	"--served-model-name", "vibevoice",
	"--trust-remote-code",
	"--dtype", "bfloat16",
	"--max-num-seqs", "16", # 64
	"--max-model-len", "4096", # 65536
	# "--max-num-batched-tokens", "32768",
	"--gpu-memory-utilization", "0.9", # 0.8
	"--enforce-eager",
	"--no-enable-prefix-caching",
	"--enable-chunked-prefill",
	"--chat-template-content-format", "openai",
	"--tensor-parallel-size", "4", # 1
	"--allowed-local-media-path", "/app",
	"--port", str(port),
]

# 修改 vllm_plugin/tests/test_api.py
payload = {
	...
	"max_tokens": 2048,       
	...
}
  
```

运行server
```sh
# 1. 先强制删除已存在的同名容器，防止 Conflict 报错
docker rm -f vibevoice-custom 2>/dev/null

# 2. 运行新容器
docker run -it --gpus all --name vibevoice-custom \
  --ipc=host \
  --rm \
  -p 8000:8000 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -v $(pwd):/app \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /app \
  --entrypoint bash \
  vllm/vllm-openai:v0.14.1 \
  -c "python3 /app/vllm_plugin/scripts/start_server.py"
```