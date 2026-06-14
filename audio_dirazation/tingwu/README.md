# 阿里云通义听悟 (Tingwu) 离线转写

使用阿里云通义听悟 Tea SDK，提交离线音视频 URL，轮询任务状态，下载转写/翻译/摘要等结果。

## 功能

- 离线音视频转写 (ASR)
- 说话人分离 (Speaker Diarization)
- 多语言翻译
- 会议纪要 / 摘要
- SRT 字幕导出 (带说话人标识)

## 环境变量 (.env)

```dotenv
ALIBABA_CLOUD_ACCESS_KEY_ID=your_access_key_id
ALIBABA_CLOUD_ACCESS_KEY_SECRET=your_access_key_secret
TINGWU_APP_KEY=your_tingwu_app_key
TINGWU_REGION_ID=cn-beijing
TINGWU_ENDPOINT=tingwu.cn-beijing.aliyuncs.com
```

## 依赖

```bash
pip install alibabacloud-tea-openapi alibabacloud-tingwu20230930 requests python-dotenv
```

## Python 用法

```python
import os
from dotenv import load_dotenv
from audio_dirazation.tingwu import TingwuClient

load_dotenv()

client = TingwuClient(
    access_key_id=os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID"),
    access_key_secret=os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET"),
    app_key=os.getenv("TINGWU_APP_KEY"),
)

# 创建离线任务
payload = client.create_offline_task(
    file_url="https://example.com/demo.mp4",
    source_language="cn",
    speaker_count=2,  # 0=自动推断
    enable_summarization=True,
)
task_id = client.extract_task_id(payload)
print(f"TaskId: {task_id}")

# 等待完成
task_info = client.wait_for_task(task_id)

# 获取结果 URL
result_urls = client.extract_result_urls(task_info)

# 下载结果文件
downloads = client.download_result_files(result_urls, output_dir="work/tingwu/downloads")

# 导出带说话人的 SRT 字幕
transcription = client.load_json("work/tingwu/downloads/transcription.json")
segments = client.extract_transcription_segments(transcription)
client.write_srt(segments, "work/tingwu/speaker_subtitles.srt")
```

## CLI 用法

```bash
python -m audio_dirazation.tingwu.cli \
  --file-url 'https://example.com/demo.mp4' \
  --source-language cn \
  --speaker-count 2 \
  --wait \
  --download-results \
  --export-srt \
  --output-dir work/tingwu
```

## 参考

- [听悟 API 概览](https://help.aliyun.com/zh/tingwu/api-tingwu-2023-09-30-overview)
- [CreateTask](https://help.aliyun.com/zh/tingwu/api-tingwu-2023-09-30-createtask)
- [GetTaskInfo](https://help.aliyun.com/zh/tingwu/api-tingwu-2023-09-30-gettaskinfo)
