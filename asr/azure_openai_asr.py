# -- coding: utf-8 --
# @Time : 2024/8/18
# @Author : ykk648
"""
pip install openai==1.41.0
ref https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/whisper-overview#whisper-model-via-azure-ai-speech-or-via-azure-openai-service
https://platform.openai.com/docs/guides/speech-to-text/quickstart
"""
from dotenv import load_dotenv

load_dotenv()
import os
from openai import AzureOpenAI


class AzureOpenAIASR:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_WHISPER_API_KEY"),
            api_version="2024-06-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_WHISPER_ENDPOINT")
        )

    def forward(self, audio_path, language='zh', srt_out_p=None):
        # https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-timestamp_granularities
        srt_result = self.client.audio.transcriptions.create(
            file=open(audio_path, "rb"),
            model='whisper',
            language=language,
            response_format="srt",  # srt verbose_json
            # timestamp_granularities=["segment"] # word segment
        )
        return srt_result
