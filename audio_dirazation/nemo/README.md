ref https://github.com/NVIDIA/NeMo/blob/main/examples/speaker_tasks/diarization/README.md

https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html

#### Install:

Don't support Windows.

```shell
git clone https://github.com/NVIDIA/NeMo/tree/a5c841312f3a909777f4726deda7d4efe2761910

conda create --name nemo python==3.10.12
conda activate nemo
# install pytorch torchvision torchaudio
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging

cd NeMo
pip install ./

pip install pytorch-lightning==2.2.1

python multiscale_diar_decoder_infer.py --config-path='../conf/inference' --config-name='diar_infer_telephonic.yaml' diarizer.out_dir=./ diarizer.manifest_filepath=./test.json
```

#### json file

test.json，**not support indent, only support 16k mano wav format audio**

```json
{"audio_filepath": "./anne-hathaway-16k-mano.wav", "offset": 0, "duration": null, "label": "infer", "text": "-", "num_speakers": null, "rttm_filepath": "./test.rttm", "uem_filepath": "./test.uem"}
```
