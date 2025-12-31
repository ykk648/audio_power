# -- coding: utf-8 --
# @Time : 2024/1/8
# @Author : ykk648

from speak_detection import SileroVAD
from whisperx import load_audio
import whisperx

audio_p = 'audio_vocals.wav'
svad = SileroVAD()
tss = svad.forward(audio_p)

print(tss)

audio_array = load_audio(audio_p)

print(audio_array.shape)

whisper_model = whisperx.load_model('large', language='zh',
                                                 download_root='pretrain_models/audio_lib/speak_recognition/whisper',
                                                 device='cuda')
# align_model, align_metadata = whisperx.load_align_model(language_code='zh', device='cuda')
for ts in tss:
    audio_partial = audio_array[ts['start']:ts['end']]
    rec_result = whisper_model.transcribe(audio_partial, batch_size=32, language='zh')
    # rec_result = whisperx.align(rec_result['segments'], align_model, align_metadata, wav_path, 'cuda', return_char_alignments=False)
    print(rec_result)