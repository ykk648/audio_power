## 在线部署（阿里云FunASR服务）

### 安装依赖

```commandline
pip install dashscope pyaudio python-dotenv
```

### 配置API Key

创建 `.env` 文件（在项目根目录），添加：

```env
DASHSCOPE_API_KEY=sk-xxx
```

### 直接运行示例

```bash
python asr/funasr_api/aliyun_funasr_api.py
```

### 相关链接

- 体验地址: https://bailian.console.aliyun.com/?tab=model#/efm/model_experience_center/voice?currentTab=voiceAsr&modelId=fun-asr-realtime
- API文档: https://help.aliyun.com/zh/model-studio/fun-asr-realtime-python-sdk
- SDK安装: https://help.aliyun.com/zh/model-studio/install-sdk?spm=a2c4g.11186623.0.0.5b6923e4etXPo0

---

## 离线部署

### SenseVoice which support offline/2pass

参考 https://github.com/modelscope/FunASR/blob/main/runtime/readme_cn.md#%E4%B8%AD%E6%96%87%E5%AE%9E%E6%97%B6%E8%AF%AD%E9%9F%B3%E5%90%AC%E5%86%99%E6%9C%8D%E5%8A%A1cpu%E7%89%88%E6%9C%AC
客户端官方代码 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/sample/funasr_samples.tar.gz

#### 部署服务端

```shell
sudo docker run -p 10096:10095 -it --privileged=true \
  -v $PWD/funasr-runtime-resources/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12

cd FunASR/runtime/

bash run_server_2pass.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir iic/SenseVoiceSmall-onnx  \
  --certfile 0 \
  --online-model-dir damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx  \
  --punc-dir damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
  --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst \
  --itn-dir thuduj12/fst_itn_zh \
  --hotword /workspace/models/hotwords.txt

# 只支持中文, 速度更快:   --model-dir damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx
```

#### 客户端连接

```
# 代码中修改服务端IP地址
python funasr_client_api.py
```