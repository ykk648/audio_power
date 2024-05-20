# -- coding: utf-8 --
# @Time : 2023/11/3
# @Author : ykk648

"""
partly borrowed from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/lib/rmvpe.py
"""
import librosa
import torch
import numpy as np


class MelSpectrogram:
    def __init__(self, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None):
        n_fft = win_length if n_fft is None else n_fft
        self.mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        # mel_basis = torch.from_numpy(mel_basis).float()
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio, center=True):
        """
        Args:
            audio: tensor (ori_frame) frame*hop_size
            center:
        Returns: tensor (channel, frame)
        """
        fft = librosa.stft(audio,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length,
                           center=center,
                           window=librosa.filters.get_window('hann', Nx=1024),
                           pad_mode='reflect')
        magnitude = np.sqrt(np.power(fft.real, 2) + np.power(fft.imag, 2))
        mel_output = np.matmul(self.mel_basis, magnitude)
        log_mel_spec = np.log(np.clip(mel_output, a_min=1e-5, a_max=None))
        return log_mel_spec

    def torch_forward(self, audio, center=True):
        """
        Args:
            audio: tensor (ori_frame (frame*hop_size))
            center:
        Returns: tensor (channel, frame)
        """
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length),
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        mel_output = torch.matmul(torch.from_numpy(self.mel_basis), magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


if __name__ == '__main__':
    audio_ = librosa.core.load("resources/guangdian.wav", sr=16000, mono=True)[0]
    ms = MelSpectrogram(n_mel_channels=128, sampling_rate=16000, win_length=1024, hop_length=160, n_fft=1024,
                        mel_fmin=30, mel_fmax=8000)
    print(ms.forward(audio_).shape)
    print(ms.torch_forward(torch.from_numpy(audio_).float()).shape)
