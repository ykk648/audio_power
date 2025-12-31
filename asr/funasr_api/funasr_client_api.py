"""
FunASR 客户端 API - 支持麦克风输入
ref https://github.com/alibaba-damo-academy/FunASR
"""
import ssl
import json
import asyncio
import websockets
import pyaudio
from queue import Queue


class FunASRMicrophoneAPI:
    def __init__(self, host="127.0.0.1", port=10095, is_ssl=True, 
                 mode="2pass", chunk_size="5,10,5", chunk_interval=10,
                 use_itn=True, hotword=""):
        """
        初始化FunASR麦克风识别客户端
        
        :param host: FunASR服务器地址
        :param port: FunASR服务器端口
        :param is_ssl: 是否使用SSL连接
        :param mode: 识别模式，可选: "online"(流式), "offline"(非流式), "2pass"(流式+非流式)
        :param chunk_size: 音频块大小，格式: "5,10,5" (单位: 100ms)
        :param chunk_interval: 发送间隔 (单位: ms)
        :param use_itn: 是否使用逆文本规范化
        :param hotword: 热词文件路径或热词字符串
        """
        self.host = host
        self.port = port
        self.is_ssl = is_ssl
        self.mode = mode
        # 处理chunk_size格式，支持 "5,10,5" 或 "5, 10, 5"
        self.chunk_size = [int(x.strip()) for x in chunk_size.split(",")]
        self.chunk_interval = chunk_interval
        self.use_itn = use_itn
        self.hotword = hotword
        
        # 音频参数
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        # 计算音频块大小（与官方示例保持一致）
        chunk_size_ms = 60 * self.chunk_size[1] / self.chunk_interval
        self.CHUNK = int(self.RATE / 1000 * chunk_size_ms)
        
        # 结果队列
        self.result_queue = Queue()
        self.is_running = False
        
    def _load_hotwords(self):
        """加载热词"""
        hotword_msg = ""
        if self.hotword.strip():
            try:
                # 如果是文件路径
                if self.hotword.endswith('.txt'):
                    fst_dict = {}
                    with open(self.hotword, 'r', encoding='utf-8') as f:
                        for line in f:
                            words = line.strip().split(" ")
                            if len(words) >= 2:
                                try:
                                    fst_dict[" ".join(words[:-1])] = int(words[-1])
                                except ValueError:
                                    pass
                    hotword_msg = json.dumps(fst_dict)
                else:
                    # 如果是字符串，格式: "热词1 权重1 热词2 权重2"
                    words = self.hotword.strip().split()
                    if len(words) >= 2 and len(words) % 2 == 0:
                        fst_dict = {}
                        for i in range(0, len(words), 2):
                            try:
                                fst_dict[words[i]] = int(words[i+1])
                            except ValueError:
                                pass
                        hotword_msg = json.dumps(fst_dict)
            except Exception as e:
                print(f"Warning: Failed to load hotwords: {e}")
        return hotword_msg
    
    async def _record_microphone(self, websocket):
        """从麦克风录音并发送到服务器"""
        p = pyaudio.PyAudio()
        stream = None
        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            # 发送初始配置消息（与官方示例格式保持一致）
            hotword_msg = self._load_hotwords()
            message = json.dumps({
                "mode": self.mode,
                "chunk_size": self.chunk_size,
                "chunk_interval": self.chunk_interval,
                "wav_name": "microphone",
                "is_speaking": True,
                "hotwords": hotword_msg,
                "itn": self.use_itn
            })
            await websocket.send(message)
            print(f"[连接成功] 开始录音，模式: {self.mode}")
            
            # 持续录音并发送（使用while True，与官方示例一致）
            while True:
                try:
                    if not self.is_running:
                        # 发送结束消息
                        try:
                            end_message = json.dumps({"is_speaking": False})
                            await websocket.send(end_message)
                        except:
                            pass
                        break
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    await websocket.send(data)
                    await asyncio.sleep(0.005)  # 短暂延迟
                except Exception as e:
                    print(f"录音错误: {e}")
                    # 发送结束消息
                    try:
                        end_message = json.dumps({"is_speaking": False})
                        await websocket.send(end_message)
                    except:
                        pass
                    break
            
        except Exception as e:
            print(f"麦克风初始化错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            try:
                p.terminate()
            except:
                pass
    
    async def _receive_messages(self, websocket):
        """接收服务器返回的识别结果"""
        text_print = ""
        text_print_2pass_online = ""
        text_print_2pass_offline = ""
        try:
            while True:
                try:
                    msg = await websocket.recv()
                    if msg is None or len(msg) == 0:
                        continue
                    
                    msg_dict = json.loads(msg)
                    self.result_queue.put(msg_dict)
                    
                    # 打印结果（与官方示例格式保持一致）
                    text = msg_dict.get("text", "")
                    if not text:
                        continue
                    
                    mode = msg_dict.get("mode", "")
                    wav_name = msg_dict.get("wav_name", "microphone")
                    timestamp = msg_dict.get("timestamp", "")
                    
                    # 处理offline模式的结束标志
                    if mode == "offline":
                        offline_msg_done = msg_dict.get("is_final", False)
                        if offline_msg_done:
                            self.is_running = False
                    
                    if 'mode' not in msg_dict:
                        continue
                    
                    if mode == "online":
                        # 流式结果，追加显示
                        text_print += "{}".format(text)
                        text_print = text_print[-10000:]  # 限制长度
                        print(f"\r[流式识别] {text_print}", end="", flush=True)
                    elif mode == "offline":
                        # 非流式结果
                        if timestamp:
                            print(f"\n[非流式识别] {wav_name}: {text} (时间戳: {timestamp})")
                        else:
                            print(f"\n[非流式识别] {wav_name}: {text}")
                    elif mode == "2pass-online":
                        # 2pass流式结果
                        text_print_2pass_online += "{}".format(text)
                        text_print = text_print_2pass_offline + text_print_2pass_online
                        text_print = text_print[-10000:]
                        print(f"\r[2pass-流式] {text_print}", end="", flush=True)
                    elif mode == "2pass-offline":
                        # 2pass非流式结果
                        text_print_2pass_online = ""
                        text_print = text_print_2pass_offline + "{}".format(text)
                        text_print_2pass_offline += "{}".format(text)
                        text_print = text_print[-10000:]
                        print(f"\n[2pass-非流式] {text_print}", flush=True)
                    else:
                        # 其他模式
                        print(f"\n[识别结果] {text}", flush=True)
                        
                except websockets.exceptions.ConnectionClosed:
                    print("\n[连接已关闭]")
                    self.is_running = False
                    break
                except Exception as e:
                    print(f"\n接收消息错误: {e}")
                    import traceback
                    traceback.print_exc()
                    self.is_running = False
                    break
        except Exception as e:
            print(f"接收消息线程错误: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False
    
    async def _run(self):
        """运行主循环"""
        # 构建连接URI
        if self.is_ssl:
            uri = f"wss://{self.host}:{self.port}"
            ssl_context = ssl.SSLContext()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        else:
            uri = f"ws://{self.host}:{self.port}"
            ssl_context = None
        
        print(f"[正在连接] {uri}")
        print(f"[连接参数] host={self.host}, port={self.port}, ssl={self.is_ssl}")
        
        # 先测试端口连通性
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            if result != 0:
                print(f"[错误] 无法连接到 {self.host}:{self.port}")
                print(f"       请检查:")
                print(f"       1. FunASR服务是否已启动")
                print(f"       2. 服务器地址和端口是否正确")
                print(f"       3. 防火墙是否允许该端口")
                return
            else:
                print(f"[端口检查] 端口 {self.port} 可访问")
        except Exception as e:
            print(f"[端口检查] 无法检查端口: {e}")
        
        try:
            # 使用与官方示例完全一致的连接参数
            async with websockets.connect(
                uri, 
                subprotocols=["binary"], 
                ping_interval=None, 
                ssl=ssl_context
            ) as websocket:
                print(f"[连接成功] WebSocket握手完成")
                self.is_running = True
                
                # 创建录音和接收消息的并发任务
                record_task = asyncio.create_task(self._record_microphone(websocket))
                receive_task = asyncio.create_task(self._receive_messages(websocket))
                
                # 等待任务完成
                try:
                    await asyncio.gather(record_task, receive_task)
                except Exception as e:
                    print(f"任务执行错误: {e}")
                    import traceback
                    traceback.print_exc()
        except websockets.exceptions.InvalidURI as e:
            print(f"[错误] 连接URI错误: {e}")
            print(f"       请检查URI格式: {uri}")
        except websockets.exceptions.InvalidMessage as e:
            print(f"[错误] WebSocket握手失败: {e}")
            print(f"       可能的原因:")
            print(f"       1. 服务器不是WebSocket服务（可能是HTTP或其他协议）")
            print(f"       2. 端口号错误（当前端口: {self.port}）")
            print(f"       3. 服务器需要SSL但客户端未启用（当前SSL: {self.is_ssl}）")
            print(f"       4. 服务器配置不正确")
            print(f"       建议: 检查FunASR服务的配置和日志")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[错误] 连接已关闭: {e}")
        except ConnectionRefusedError as e:
            print(f"[错误] 连接被拒绝: {e}")
            print(f"       请检查FunASR服务是否已启动，地址: {uri}")
        except OSError as e:
            print(f"[错误] 网络错误: {e}")
            print(f"       请检查网络连接和服务器地址")
        except Exception as e:
            print(f"[错误] 连接错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
    
    def start(self):
        """启动识别（阻塞）"""
        try:
            asyncio.run(self._run())
        except KeyboardInterrupt:
            print("\n[用户中断] 停止识别")
            self.is_running = False
        except Exception as e:
            print(f"运行错误: {e}")
            self.is_running = False
    
    def stop(self):
        """停止识别"""
        self.is_running = False
    
    def get_results(self):
        """获取识别结果队列中的所有结果"""
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results


if __name__ == '__main__':
    # 使用示例
    print("FunASR 麦克风识别客户端")
    print("=" * 50)
    
    # 创建客户端实例
    client = FunASRMicrophoneAPI(
        host="192.168.1.1",  # 修改为你的FunASR服务器地址
        port=10096,        # 修改为你的FunASR服务器端口
        is_ssl=False,      # 根据服务器配置修改
        mode="2pass",      # 可选: "online", "offline", "2pass"
        chunk_size="5,10,5",
        chunk_interval=10,
        use_itn=True,
        # hotword="",  # 可选: 热词文件路径或热词字符串
    )
    
    # 启动识别
    print("\n提示: 按 Ctrl+C 停止识别\n")
    client.start()

