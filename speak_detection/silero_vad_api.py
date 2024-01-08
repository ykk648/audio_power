# -- coding: utf-8 --
# @Time : 2023/12/5
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import torch
import os

torch.set_num_threads(1)
os.environ['ORT_LOGGING_LEVEL'] = '1'

SAMPLING_RATE = 16000
MODEL_DIR = 'pretrain_models/audio_lib/speak_detect/snakers4_silero-vad_master'


class SileroVAD:
    def __init__(self, return_seconds=False):
        self.model, utils = torch.hub.load(
            repo_or_dir=MODEL_DIR,
            source='local',
            model='silero_vad',
            force_reload=False,
            force_onnx_cpu=True,
            onnx=True)

        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks,
         self.drop_chunks) = utils

        self.wav = None
        self.speech_timestamps = None
        self.return_seconds = return_seconds

    def forward(self, audio_path, save_chunks=False, return_mute=False):
        self.wav = self.read_audio(audio_path, sampling_rate=SAMPLING_RATE)
        # get speech timestamps from full audio file
        self.speech_timestamps = self.get_speech_timestamps(self.wav, self.model, window_size_samples=1536,
                                                            return_seconds=self.return_seconds,
                                                            sampling_rate=SAMPLING_RATE)

        if save_chunks:
            assert not self.return_seconds
            # merge all speech chunks to one audio
            self.save_audio('only_speech.wav',
                            self.collect_chunks(self.speech_timestamps, self.wav), sampling_rate=SAMPLING_RATE)

            self.save_audio('drop_chunks.wav',
                            self.drop_chunks(self.speech_timestamps, self.wav), sampling_rate=SAMPLING_RATE)
        if return_mute:
            return self.get_mute_timestamps()

        return self.speech_timestamps

    @staticmethod
    def timestamps2frame(speech_timestamps, fps):
        result_ids = []
        for i in speech_timestamps:
            start = i['start']
            end = i['end']
            start_id = int((start / SAMPLING_RATE) * fps)
            end_id = int((end / SAMPLING_RATE) * fps)
            result_ids.append([start_id, end_id])
            if start_id >= end_id:
                continue
        return result_ids

    def get_mute_timestamps(self):
        wav_len = len(self.wav)
        result_ids = []
        cur = 0
        for i in self.speech_timestamps:
            result_ids.append({'start': cur, 'end': i['start'] - 1})
            cur = i['end'] + 1

        result_ids.append({'start': cur, 'end': wav_len})
        return result_ids


if __name__ == '__main__':
    audio_p = 'resources/audio/chinese_poem2.wav'
    silero = SileroVAD(return_seconds=True)
    print(silero.forward(audio_p))

    silero = SileroVAD(return_seconds=False)
    ts = silero.forward(audio_p)
    print(ts)
    print(silero.timestamps2frame(ts, fps=30))

    mst = silero.forward(audio_p, return_mute=True)
    print(mst)
    print(silero.timestamps2frame(mst, fps=30))

    """
    [{'start': 0.2, 'end': 2.9}, {'start': 3.4, 'end': 4.5}, {'start': 4.8, 'end': 6.1}, {'start': 6.7, 'end': 7.1}, {'start': 7.4, 'end': 7.8}, {'start': 8.1, 'end': 9.2}]
    [{'start': 2592, 'end': 46560}, {'start': 54816, 'end': 72672}, {'start': 76320, 'end': 97248}, {'start': 107040, 'end': 114144}, {'start': 117792, 'end': 124896}, {'start': 130080, 'end': 147936}]
    [[4, 87], [102, 136], [143, 182], [200, 214], [220, 234], [243, 277]]
    [{'start': 0, 'end': 2591}, {'start': 46561, 'end': 54815}, {'start': 72673, 'end': 76319}, {'start': 97249, 'end': 107039}, {'start': 114145, 'end': 117791}, {'start': 124897, 'end': 130079}, {'start': 147937, 'end': 153600}]
    [[0, 4], [87, 102], [136, 143], [182, 200], [214, 220], [234, 243], [277, 288]]
    """
