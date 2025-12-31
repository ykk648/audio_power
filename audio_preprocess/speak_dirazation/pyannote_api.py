# -- coding: utf-8 --
# @Time : 2024/11/1
# @Author : ykk648
"""
pip install pyannote.audio
"""
import os
from dotenv import load_dotenv

load_dotenv()
from pyannote.audio import Pipeline

# first init needs HF token
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("HF_TOKEN"))

# send pipeline to GPU (when available)
import torch

pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

with ProgressHook() as hook:
    # 不支持mp3
    diarization = pipeline(r".\ai-podcast-denoise-16k-mono.wav", min_speakers=3, max_speakers=3, hook=hook)

# 打印结果并保存为RTTM格式
rttm_output = "output.rttm"  # 设置输出文件名

with open(rttm_output, 'w') as f:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # 格式化RTTM行
        rttm_line = f"SPEAKER 0 {turn.start:.1f} {turn.end - turn.start:.1f} <NA> <NA> {speaker} <NA>\n"
        f.write(rttm_line)

print(f"RTTM data saved to {rttm_output}")