# -- coding: utf-8 --
# @Time : 2023/11/2
# @Author : ykk648

"""
partly borrowed from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/lib/rmvpe.py
"""
from apstone import ModelBase
import numpy as np
import torch
from cv2box import CVFile

"""
input_name:['mel'], 
shape:[[1, 'mel_dynamic_axes_1', 128]], 
type:['tensor(float)']
output_name:['f0'], 
shape:[[1, 'f0_dynamic_axes_1', 360]], 
type:['tensor(float)']
"""
MODEL_ZOO = {
    'fcpe_onnx': {
        # 'model_path': 'pretrain_models/audio_lib/rmvpe/fcpe.onnx'
        'model_path': 'pretrain_models/audio_lib/rmvpe/fcpe_cents_include.onnx'
    },
}


class FCPE(ModelBase):
    def __init__(self, model_type='fcpe_onnx', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        self.input_std = self.input_mean = 127.5
        self.input_size = (112, 112)

        # cents 音分
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def to_local_average_cents(self, salience, thred=0.05):
        """
        局部平均音高（local average cents）
        Args:
            salience:
            thred:
        Returns:
        """
        center = np.argmax(salience, axis=1)  # 帧长#index
        salience = np.pad(salience, ((0, 0), (4, 4)))  # 帧长,368
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx]: ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx]: ends[idx]])
        todo_salience = np.array(todo_salience)  # 帧长，9
        todo_cents_mapping = np.array(todo_cents_mapping)  # 帧长，9
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)  # 帧长
        devided = product_sum / weight_sum  # 帧长
        maxx = np.max(salience, axis=1)  # 帧长
        devided[maxx <= thred] = 0
        return devided

    def forward(self, audio_mel, thred=0.3):
        """
        Args:
            audio_mel: (128, frame)
        Returns: (frame)
        """
        n_frames = audio_mel.shape[-1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            audio_mel = np.pad(audio_mel, ((0, 0), (0, 0), (0, n_pad)), mode="constant", constant_values=0)
        hidden = self.model.forward(audio_mel.transpose((0, 2, 1)))[0][0]

        # # decode
        # cents_pred = self.to_local_average_cents(hidden[:n_frames, :], thred=thred)
        # # cents_pred = self.latent2cents_local_decoder(hidden[:n_frames, :][None, ...], threshold=thred)
        # # f0 = np.array([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred])
        # f0 = 10 * (2 ** (cents_pred / 1200))
        # f0[f0 == 10] = 0

        return hidden
        # return f0[:n_frames, :]


if __name__ == '__main__':
    import librosa
    from audio_feature.mel import MelSpectrogram

    audio_path = 'resources/guangdian.wav'

    audio_all = librosa.load("resources/guangdian.wav", sr=16000, mono=True)[0]

    from cv2box import MyFpsCounter

    ms = MelSpectrogram(n_mel_channels=128, sampling_rate=16000, win_length=1024, hop_length=160, n_fft=1024,
                        mel_fmin=30, mel_fmax=8000)
    fcpe = FCPE(model_type='fcpe_onnx', provider='gpu')
    # fps: 11.8 GPU: 5.3G-1G
    with MyFpsCounter() as mfc:
        for _ in range(100):
            mel = ms.forward(audio_all)
            f0 = fcpe.forward(mel[None, ...], 0.3)
            # print(f0)
        print(mel.shape)
        print(f0.shape)
