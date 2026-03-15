# audio_power

音频相关能力集合：ASR、TTS、语音转换、人声分离等。

## 结构概览

- **asr**：语音识别（Azure、SenseVoice、VibeVoice、Faster-Whisper、FunASR 等）
- **tts**：语音合成（Azure、Index TTS、CosyVoice 等）
- **vc**：语音转换（RVC、SVC 等）
- **audio_preprocess**：人声分离、VAD、说话人分割等
- **audio_translation**：翻译相关
- **examples**：示例脚本

## 环境与密钥

- Python 依赖：`pip install -r requirements.txt`，各子模块可能有单独 `requirements.txt`。
- **密钥与配置**：通过环境变量配置，请勿将真实 Key 写入代码。可复制 `.env.example` 为 `.env` 并填写（`.env` 已加入 `.gitignore`）。

## 使用示例

- VibeVoice ASR：`examples/test_vibevoice.py`，参见 `asr/vibevoice/README.md`。
- 其他模块见各子目录下的 README。
