"""
ref https://github.com/ina-foss/inaSpeechSegmenter
"""
from audio_preprocess.age_gender.w2v2_age_gender import AgeGender
import librosa

age_gender_model = AgeGender()

audio_path = 'resources/Obama_audio.wav'
input_audio = librosa.load(audio_path, sr=16000, mono=True)[0][None, ...][:160000]
segmentation = age_gender_model.forward(input_audio)
print(segmentation)