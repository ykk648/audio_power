## Index TTS API

基于 [index-tts-vllm](https://github.com/Ksuriuri/index-tts-vllm) 的 TTS API 封装。

### 服务端

ref [index-tts-vllm](https://github.com/Ksuriuri/index-tts-vllm)
1.5速度快于2

```sh
# install
git clone https://github.com/Ksuriuri/index-tts-vllm.git
cd index-tts-vllm
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate

# download models
modelscope download --model kusuriuri/Index-TTS-vLLM --local_dir ./checkpoints/Index-TTS-vLLM
modelscope download --model kusuriuri/Index-TTS-1.5-vLLM --local_dir ./checkpoints/Index-TTS-1.5-vLLM
modelscope download --model kusuriuri/IndexTTS-2-vLLM --local_dir ./checkpoints/IndexTTS-2-vLLM

# webui
python webui_v2.py
# api
python api_server.py --model_dir ./checkpoints/Index-TTS-1.5-vLLM --port 11996
```

### 安装依赖

```bash
pip install requests
```

### 使用方法

```python
from tts.index_tts_api.index_tts_api import IndexTTSAPI

# 创建API客户端
api = IndexTTSAPI(server_host="0.0.0.0", server_port=6006)

# 方式1: 使用本地音频文件作为参考（支持多参考音频）
api.tts_with_audio_paths(
    text="还是会想你，还是想登你",
    audio_paths=["assets/jay_promptvn.wav", "assets/vo_card_klee_endOfGame_fail_01.wav"],
    output_path="output.wav"
)

# 方式2: 使用角色名（需要在服务器端注册）
api.tts_with_character(
    text="还是会想你，还是想登你",
    character="jay_klee",
    output_path="output.wav"
)
```

### API 说明

#### `IndexTTSAPI` 类

**初始化参数:**
- `server_host` (str, 可选): TTS服务器地址，默认 "0.0.0.0"
- `server_port` (int, 可选): TTS服务器端口，默认 6006

#### `tts_with_audio_paths` 方法

使用本地音频文件作为参考音频进行TTS合成。

**参数:**
- `text` (str): 要合成的文本
- `audio_paths` (str | List[str]): 参考音频文件路径，可以是单个路径或路径列表（支持多参考音频）
- `output_path` (str): 输出音频文件路径

**返回:**
- `bool`: 成功返回True，失败返回False

#### `tts_with_character` 方法

使用角色名进行TTS合成。

**参数:**
- `text` (str): 要合成的文本
- `character` (str): 角色名称（需要在服务器端注册，参考 `assets/speaker.json`）
- `output_path` (str): 输出音频文件路径

**返回:**
- `bool`: 成功返回True，失败返回False

### 相关链接

- 项目地址: https://github.com/Ksuriuri/index-tts-vllm
- API示例: https://github.com/Ksuriuri/index-tts-vllm/blob/master/api_example.py

