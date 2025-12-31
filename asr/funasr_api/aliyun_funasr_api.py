"""
阿里云FunASR实时语音识别API - 支持麦克风输入
ref https://help.aliyun.com/zh/model-studio/fun-asr-realtime-python-sdk
"""
import os
import pyaudio
import threading
from queue import Queue
from dotenv import load_dotenv
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback

# 加载环境变量
load_dotenv()

# 设置API Key
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

# 设置WebSocket API URL（默认北京地域，如需使用新加坡地域，取消下面注释）
# dashscope.base_websocket_api_url = 'wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference'


class RecognitionCallbackHandler(RecognitionCallback):
    """识别结果回调处理器"""
    
    def __init__(self, result_queue, api_instance):
        """
        初始化回调处理器
        
        :param result_queue: 结果队列，用于存储识别结果
        :param api_instance: AliyunFunASRMicrophoneAPI实例，用于访问相关方法
        """
        self.result_queue = result_queue
        self.api_instance = api_instance
    
    def on_open(self):
        """连接建立时的回调"""
        print("[连接成功] 已连接到阿里云FunASR服务")
    
    def on_event(self, event):
        """
        接收到识别结果时的回调
        
        :param event: 识别事件对象
        """
        try:
            # 检查event对象的结构
            if not event:
                return
            
            # 调试：打印event的实际结构（首次调用时）
            if not hasattr(self, '_debug_printed'):
                print(f"[调试] event类型: {type(event)}")
                if hasattr(event, '__dict__'):
                    print(f"[调试] event属性: {list(event.__dict__.keys())}")
                elif isinstance(event, dict):
                    print(f"[调试] event键: {list(event.keys())}")
                self._debug_printed = True
            
            # 首先尝试获取event的type属性，判断事件类型
            event_type = None
            try:
                if hasattr(event, 'type'):
                    event_type = event.type
                elif isinstance(event, dict):
                    event_type = event.get('type')
            except:
                pass
            
            # 根据文档，event.result 是 RecognitionResult 对象
            # 但需要安全地访问
            result = None
            try:
                if hasattr(event, 'result'):
                    result = event.result
                elif isinstance(event, dict):
                    # 安全地访问字典的result键
                    result = event.get('result')
            except (AttributeError, KeyError) as e:
                # 如果访问result失败，尝试其他方式
                pass
            
            # 如果无法获取result，尝试将event本身作为result
            if result is None:
                if hasattr(event, 'get_sentence'):
                    result = event
                elif isinstance(event, dict):
                    result = event
            
            # 处理识别结果
            if result:
                # 如果result有get_sentence方法，说明是RecognitionResult对象
                if hasattr(result, 'get_sentence'):
                    try:
                        sentence = result.get_sentence()
                        if sentence:
                            # 确保sentence是字典格式
                            if not isinstance(sentence, dict):
                                # 如果是对象，转换为字典
                                sentence = {
                                    'text': getattr(sentence, 'text', ''),
                                    'begin_time': getattr(sentence, 'begin_time', 0),
                                    'end_time': getattr(sentence, 'end_time', 0),
                                    'words': getattr(sentence, 'words', [])
                                }
                            
                            self.result_queue.put(sentence)
                            
                            # 打印识别结果
                            text = sentence.get('text', '')
                            if text:
                                begin_time = sentence.get('begin_time', 0)
                                end_time = sentence.get('end_time', 0)
                                is_end = False
                                if hasattr(result, 'is_sentence_end'):
                                    try:
                                        is_end = result.is_sentence_end(sentence)
                                    except:
                                        pass
                                
                                status = "[完整句子]" if is_end else "[实时识别]"
                                print(f"{status} {text} (时间: {begin_time}ms - {end_time}ms)")
                    except Exception as e:
                        print(f"[警告] 处理RecognitionResult时出错: {e}")
                # 如果result是字典，直接处理
                elif isinstance(result, dict):
                    text = result.get('text', '')
                    if text:
                        self.result_queue.put(result)
                        begin_time = result.get('begin_time', 0)
                        end_time = result.get('end_time', 0)
                        print(f"[识别结果] {text} (时间: {begin_time}ms - {end_time}ms)")
                else:
                    # 尝试直接访问result的属性
                    text = getattr(result, 'text', None)
                    if text:
                        self.result_queue.put({'text': text})
                        print(f"[识别结果] {text}")
        except Exception as e:
            # 输出更详细的错误信息用于调试
            print(f"[警告] 处理识别结果时出错: {e}")
            print(f"[调试] event类型: {type(event)}")
            # 尝试安全地打印event内容
            try:
                if hasattr(event, '__dict__'):
                    print(f"[调试] event属性: {list(event.__dict__.keys())}")
                elif isinstance(event, dict):
                    print(f"[调试] event键: {list(event.keys())}")
            except:
                pass
    
    def on_error(self, error):
        """发生错误时的回调"""
        print(f"[错误] 识别过程出错: {error}")
        if self.api_instance:
            self.api_instance.is_running = False
    
    def on_close(self):
        """连接关闭时的回调"""
        print("[连接关闭] 识别连接已关闭")


class AliyunFunASRMicrophoneAPI:
    """阿里云FunASR实时语音识别 - 麦克风输入"""
    
    def __init__(self, 
                 model='fun-asr-realtime',
                 format='pcm',
                 sample_rate=16000,
                 region='beijing'):
        """
        初始化阿里云FunASR麦克风识别客户端
        
        :param model: 模型名称，默认 'fun-asr-realtime'
        :param format: 音频格式，默认 'pcm'，支持: pcm、wav、mp3、opus、speex、aac、amr
        :param sample_rate: 采样率，默认 16000 Hz
        :param region: 地域，'beijing' 或 'singapore'，默认 'beijing'
        """
        if not dashscope.api_key:
            raise ValueError("未找到DASHSCOPE_API_KEY环境变量，请检查.env文件或环境变量配置")
        
        # 根据地域设置WebSocket URL
        if region == 'singapore':
            dashscope.base_websocket_api_url = 'wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference'
        else:
            dashscope.base_websocket_api_url = 'wss://dashscope.aliyuncs.com/api-ws/v1/inference'
        
        self.model = model
        self.format = format
        self.sample_rate = sample_rate
        
        # 音频参数
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = sample_rate
        self.CHUNK = 1600  # 每次读取100ms的音频数据 (16000 * 0.1)
        
        # 识别结果队列
        self.result_queue = Queue()
        self.is_running = False
        
        # Recognition实例
        self.recognition = None
        
        # 音频流
        self.audio_stream = None
        self.pyaudio_instance = None
        
        # 创建回调处理器
        self.callback_handler = RecognitionCallbackHandler(self.result_queue, self)
    
    def _read_audio_from_microphone(self):
        """从麦克风读取音频数据并发送到识别服务"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.audio_stream = self.pyaudio_instance.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            print(f"[开始录音] 采样率: {self.RATE}Hz, 格式: {self.format}")
            print("[提示] 按 Ctrl+C 停止识别\n")
            
            # 持续读取音频并发送
            while self.is_running:
                try:
                    audio_data = self.audio_stream.read(self.CHUNK, exception_on_overflow=False)
                    if self.recognition:
                        self.recognition.send_audio_frame(audio_data)
                except Exception as e:
                    if self.is_running:
                        print(f"[错误] 读取音频数据失败: {e}")
                    break
                    
        except Exception as e:
            print(f"[错误] 麦克风初始化失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup_audio()
    
    def _cleanup_audio(self):
        """清理音频资源"""
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass
            self.pyaudio_instance = None
    
    def start(self):
        """
        启动识别（阻塞）
        
        开始从麦克风录音并进行实时语音识别
        """
        if self.is_running:
            print("[警告] 识别已在运行中")
            return
        
        try:
            # 创建Recognition实例，传入回调处理器
            self.recognition = Recognition(
                model=self.model,
                format=self.format,
                sample_rate=self.sample_rate,
                callback=self.callback_handler
            )
            
            # 启动识别连接
            self.recognition.start()
            
            self.is_running = True
            
            # 在单独线程中读取麦克风音频
            audio_thread = threading.Thread(target=self._read_audio_from_microphone, daemon=True)
            audio_thread.start()
            
            # 等待线程结束（阻塞）
            audio_thread.join()
            
        except KeyboardInterrupt:
            print("\n[用户中断] 停止识别")
            self.stop()
        except Exception as e:
            print(f"[错误] 识别过程出错: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
    
    def stop(self):
        """停止识别"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 清理音频资源
        self._cleanup_audio()
        
        # 关闭识别连接
        if self.recognition:
            try:
                self.recognition.stop()
            except:
                pass
            self.recognition = None
        
        print("[已停止] 识别已停止")
    
    def get_results(self):
        """
        获取识别结果队列中的所有结果
        
        :return: 识别结果列表，每个结果包含 text, begin_time, end_time, words 等信息
        """
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results
    
    def get_latest_result(self):
        """
        获取最新的识别结果
        
        :return: 最新的识别结果字典，如果队列为空则返回None
        """
        latest = None
        while not self.result_queue.empty():
            latest = self.result_queue.get()
        return latest


if __name__ == '__main__':
    # 使用示例
    print("=" * 60)
    print("阿里云FunASR实时语音识别 - 麦克风输入")
    print("=" * 60)
    
    # 检查API Key
    if not dashscope.api_key:
        print("\n[错误] 未找到DASHSCOPE_API_KEY环境变量")
        print("请创建.env文件并添加:")
        print("DASHSCOPE_API_KEY=sk-xxx")
        exit(1)
    
    # 创建识别客户端
    client = AliyunFunASRMicrophoneAPI(
        model='fun-asr-realtime',  # 使用稳定版模型
        format='pcm',              # PCM格式
        sample_rate=16000,         # 16kHz采样率
        region='beijing'           # 北京地域，如需使用新加坡改为 'singapore'
    )
    
    # 启动识别
    try:
        client.start()
    except KeyboardInterrupt:
        print("\n[用户中断]")
    finally:
        client.stop()
        
        # 显示所有识别结果
        results = client.get_results()
        if results:
            print(f"\n[识别结果汇总] 共识别到 {len(results)} 个句子:")
            for i, result in enumerate(results, 1):
                text = result.get('text', '')
                begin_time = result.get('begin_time', 0)
                end_time = result.get('end_time', 0)
                print(f"  {i}. {text} ({begin_time}ms - {end_time}ms)")

