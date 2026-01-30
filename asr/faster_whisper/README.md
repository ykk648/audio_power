### Intro

#### env

```shell
# pip install faster-whisper==1.0.3
pip install "faster-whisper @ https://github.com/SYSTRAN/faster-whisper/archive/refs/heads/master.tar.gz"
# for cuda-11.8,   4.3.1 may works
# pip install --force-reinstall ctranslate2==3.24.0 
```

model will be download to `C:\Users\YOURNAME\.cache\huggingface\hub\`


```python
_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    # only support en
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}
```

#### bug fix

- libiomp5md.dll bug

    ref [issue](https://github.com/SYSTRAN/faster-whisper/issues/752#issuecomment-2146571426)

- windows cache-system warning
  
  run python as an Adminstrator , ref [issue](https://github.com/huggingface/huggingface_hub/issues/2179#issuecomment-2031412214)