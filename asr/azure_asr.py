# -- coding: utf-8 --
# @Time : 2024/7/21
# @Author : ykk648
"""
pip install azure-cognitiveservices-speech==1.38.0
"""
from dotenv import load_dotenv

load_dotenv()
import azure.cognitiveservices.speech as speechsdk
import os
import json
from pathlib import Path
from .utils import split_thres, SegmentDummy, language_dict


class AzureASR:
    def __init__(self):
        self.speech_config = speechsdk.SpeechConfig(subscription=os.getenv('AZURE_ASR_KEY'), region=os.getenv('AZURE_ASR_LOCATION'))
        self.speech_config.request_word_level_timestamps()

    def forward(self, audio_path, language='zh', word_timestamp=True):
        """
        :param audio_path:
        :param language:
        :param word_timestamp: whether to use word timestamps or not
        :return:
        """
        assert Path(audio_path).exists()
        # ref https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/language-support?tabs=stt#supported-languages
        self.speech_config.speech_recognition_language = language_dict[language]
        audio_config = speechsdk.AudioConfig(filename=audio_path)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)

        recon_segments = []
        while True:
            result = speech_recognizer.recognize_once_async().get()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # print("Recognized: {}".format(result.text))

                if len(result.text) <= split_thres[language] or not word_timestamp:
                    recon_segments.append(SegmentDummy(start=result.offset / 10000000, end=(result.duration + result.offset) / 10000000, text=result.text.lstrip()))
                else:
                    cur = None
                    for word in json.loads(result.json)['NBest'][0]['Words']:
                        word_start = word['Offset'] / 10000000
                        word_end = (word['Offset'] + word['Duration']) / 10000000
                        word_text = word['Word']
                        if not cur:
                            cur = SegmentDummy(start=word_start, end=word_end, text=word_text)
                            continue
                        if word_start - cur.end < 0.5:
                            cur.end = word_end
                            # 单词之间有空格
                            if language in ['en']:
                                cur.text += f" {word_text}"
                            else:
                                cur.text += f"{word_text}"
                            continue
                        cur.text += f"."
                        recon_segments.append(cur)
                        cur = SegmentDummy(start=word_start, end=word_end, text=word_text)
                    if cur:
                        cur.text += f"."
                        recon_segments.append(cur)
                        # full_txt.append(f"{line_count}\n{convert_seg(cur)}")
                        # line_count += 1
                print("Recognized: {}".format(result.text))
                print("Offset in Ticks: {}".format(result.offset))
                print("Duration in Ticks: {}".format(result.duration))
            elif result.reason == speechsdk.ResultReason.NoMatch:
                print("No speech could be recognized: {}".format(result.no_match_details))
                break
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print("Error details: {}".format(cancellation_details.error_details))
                    print("Did you set the speech resource key and region values?")
                break
        return recon_segments
