# -- coding: utf-8 --
# @Time : 2024/1/4
# @Author : ykk648
# @Project : https://github.com/ykk648/audio_lib
"""
ref https://github.com/m-bain/whisperX
https://github.com/liuzhao1225/YouDub-webui/blob/master/youdub/step020_whisperx.py

pip install git+https://github.com/m-bain/whisperx.git --upgrade
"""
import whisperx
import json
import os
import librosa
import numpy as np
from scipy.io import wavfile
from dotenv import load_dotenv

load_dotenv()


def save_wav(wav: np.ndarray, output_path: str, sample_rate=24000):
    # wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav * 32767
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))


def merge_segments(transcript, ending='!"\').:;?]}~'):
    merged_transcription = []
    buffer_segment = None

    for segment in transcript:
        if buffer_segment is None:
            buffer_segment = segment
        else:
            # Check if the last character of the 'text' field is a punctuation mark
            if buffer_segment['text'][-1] in ending:
                # If it is, add the buffered segment to the merged transcription
                merged_transcription.append(buffer_segment)
                buffer_segment = segment
            else:
                # If it's not, merge this segment with the buffered segment
                buffer_segment['text'] += ' ' + segment['text']
                buffer_segment['end'] = segment['end']

    # Don't forget to add the last buffered segment
    if buffer_segment is not None:
        merged_transcription.append(buffer_segment)

    return merged_transcription


def generate_speaker_audio(folder, transcript):
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    audio_data, samplerate = librosa.load(wav_path, sr=24000)
    speaker_dict = dict()
    length = len(audio_data)
    delay = 0.05
    for segment in transcript:
        start = max(0, int((segment['start'] - delay) * samplerate))
        end = min(int((segment['end'] + delay) * samplerate), length)
        speaker_segment_audio = audio_data[start:end]
        speaker_dict[segment['speaker']] = np.concatenate((speaker_dict.get(
            segment['speaker'], np.zeros((0,))), speaker_segment_audio))

    speaker_folder = os.path.join(folder, 'SPEAKER')
    if not os.path.exists(speaker_folder):
        os.makedirs(speaker_folder)

    for speaker, audio in speaker_dict.items():
        speaker_file_path = os.path.join(
            speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path)


class Whisper:
    def __init__(self):
        self.whisper_model = whisperx.load_model('large', language='zh',
                                                 download_root='pretrain_models/audio_lib/speak_recognition/whisper',
                                                 device='cuda')
        self.align_model, self.align_metadata = whisperx.load_align_model(language_code='zh', device='cuda')
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv('HF_TOKEN'), device='cuda')
        print('init done')

    def forward(self, wav_folder):
        wav_path = os.path.join(wav_folder, 'audio_vocals.wav')

        rec_result = self.whisper_model.transcribe(wav_path, batch_size=32, language='zh')
        # print(rec_result)
        print('transcribe done')
        rec_result = whisperx.align(rec_result['segments'], self.align_model, self.align_metadata,
                                    wav_path, 'cuda', return_char_alignments=False)
        print('align done')
        diarization = True
        if diarization:
            diarize_segments = self.diarize_model(wav_path, num_speakers=3, min_speakers=3, max_speakers=3)
            # print(diarize_segments)
            rec_result = whisperx.assign_word_speakers(diarize_segments, rec_result)
            print('diarization done')

        transcript = [{'start': segement['start'], 'end': segement['end'], 'text': segement['text'].strip(),
                       'speaker': segement.get('speaker', 'SPEAKER_00')} for segement in rec_result['segments']]
        transcript = merge_segments(transcript)
        with open('transcript.json', 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=4, ensure_ascii=False)
        print(f'Transcribed {wav_path} successfully, and saved to {"transcript.json"}')
        generate_speaker_audio(wav_folder, transcript)


if __name__ == '__main__':
    # audio_vocals.wav in it
    wav_folder = './'
    whis = Whisper()
    whis.forward(wav_folder)
