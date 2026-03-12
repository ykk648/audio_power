### intro

24.11 起支持基于 CTC 对齐的 timestamp，已接入：`forward(..., word_timestamp=True)` 会传入 `output_timestamp=True` 并解析返回的 `timestamp` 字段（参考 [demo2.py](https://github.com/FunAudioLLM/SenseVoice/blob/main/demo2.py)）。

#### 下载模型

```py
from modelscope import snapshot_download
model_dir = snapshot_download('iic/SenseVoiceSmall', local_dir='pretrain_models/audio_lib/speak_recognition/damo/SenseVoiceSmall')
```