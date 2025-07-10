import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QPushButton,QVBoxLayout, QWidget,QGridLayout,QListWidget,QGraphicsPixmapItem,QGraphicsScene,QDesktopWidget,QListWidgetItem,QShortcut,QMessageBox,QLineEdit,QSizePolicy,QLabel,QDialog,QDialogButtonBox,QStyledItemDelegate,QTableView,QGraphicsView
from PyQt5.QtGui import  QPainter, QPixmap,QImage,QBrush,QColor,QFont,QFontMetrics,QKeySequence,QMovie,QTransform,QIcon,QTouchEvent
from PyQt5.QtCore import QUrl,QThread,pyqtSignal,Qt,QEvent,QSize,QRect,QTimer,QTime,QDate,QDateTime,Qt,QCoreApplication
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5 import QtCore,QtGui
from flask import Flask, request, jsonify
import subprocess
import pandas as pd
import socket
import time
import wave
import pyaudio
from aip import AipSpeech
import numpy as np
import json
from urllib import request as urllib_request, parse
import pygame
import os
import requests
import serial
import threading
import struct  # 用于解析二进制数据
import re  # 导入正则表达式模块
from gui_robot import Ui_MainWindow_robot
from gui_map import Ui_SubWindow_map

#from PyQt5.QtMultimediaWidgets import QVideoWidget
# /home/orangepi/swy/cv/rknn3588-yolov8/encode.py

APP_ID = '63343045'
API_KEY = '6nVVrlWvtbumSbY2SbEdu3DB'
SECRET_KEY = 'tqekkt6M8O7PX9kIUTMx2WzzczEectJO'
CHUNK = 1024 
FORMAT = pyaudio.paInt16 # 16位深
CHANNELS = 1 #1是单声道，2是双声道。
RATE = 16000 # 采样率，调用API一般为8000或16000
RECORD_SECONDS = 10 # 录制时间10s
THRESHOLD = 1500  # 音频能量阈值，根据实际情况调整
SILENCE_TIMEOUT = 2  # 静音超时时间（秒）

SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
TIMEOUT = 1

# 接口地址
POST_URL = "http://192.168.31.220:7098/robotReceive/receiveSoundMsg"  # 替换为目标接口地址http://www.hihust.com:7098/robotReceive/receiveSoundMsg  http://www.hihust.com:28099/gsshApi/robotReceive/receiveSoundMsg

# 全局定义 recognition_thread
recognition_thread = None

 ##############语音交互线程##################
class VoiceRecognitionThread(QThread):
    audio_finished = pyqtSignal()

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.stopped = False
        self.pygame_initialized = False
        self.is_listening_for_wakeup = True  # 控制是否在等待唤醒命令
        self.serial_thread = threading.Thread(target=self.listen_for_wakeup_command)
        self.serial_thread.daemon = True  # 设为守护线程，主线程退出时它会自动退出

    def run(self):
        self.serial_thread.start()  # 启动串口监听线程
        self.exec_()  # 启动Qt事件循环，继续处理GUI交互等
    
    def listen_for_wakeup_command(self):
        # 实现唤醒命令监听逻辑
        print('监听唤醒命令...')
        while self.is_listening_for_wakeup:  # 控制监听状态
            #有一个串口监听函数 `read_from_serial()` 用于读取数据
            data = self.read_from_serial()
            if data == "01":
                self.respond_to_wakeup_command()
                self.conversation()  # 如果没有接收到其他有效命令，进入语音识别
            else:
                print("没有唤醒命令")
            time.sleep(0.1)  # 加点延时，防止CPU占用过高
    
    def respond_to_wakeup_command(self):
        response_text = "我在。"
        self.speech_synthesis(response_text)  # 语音合成
        self.play_audio("/home/orangepi/python/gui/result.wav")  # 播放“我在”的音频

    def read_from_serial(self):
        # try:
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE) as ser:
                print(f"串口 {SERIAL_PORT} 打开成功，波特率: {BAUD_RATE}")
                while True:
                    data = ser.read(1)  # 读取1个字节
                    if data:
                        print(f"接收到的数据: {data.hex()}")  # 十六进制格式
                        return data.hex()
                    # else:
                    #     # self.talking_received.emit()
                    #     print("没有接收到数据")
        # except serial.SerialException as e:
        #     print(f"串口错误: {e}")

    def save_wave_file(self, pa, data):
        with wave.open(self.filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(data))

    def detect_sound(self, data):
        energy = np.sum(np.frombuffer(data, dtype=np.int16)**2) / len(data)
        return energy > THRESHOLD

    def get_audio_and_recognize(self):
        print('开始语音识别...')
        time.sleep(1)
        try:
            # input_device_index = self.get_device_index_by_name("rockchip-es8388", input=True)
            pa = pyaudio.PyAudio()
            stream = pa.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            # input_device_index=input_device_index,
                            frames_per_buffer=CHUNK)
            
        except OSError as e:
            print("无法找到音频输入设备，检查系统设置和设备连接。")
            return ""

        frames = []
        started = False
        start_time = time.time()

        while not self.stopped:
            data = stream.read(CHUNK)
            frames.append(data)
            
            if self.detect_sound(data):
                if not started:
                    started = True
                start_time = time.time()
            elif started:
                if time.time() - start_time > SILENCE_TIMEOUT:
                    break

        stream.stop_stream()
        stream.close()
        pa.terminate()

        self.save_wave_file(pa, frames)
        
        client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
        result = client.asr(self.get_file_content(), 'wav', RATE, {'dev_pid': 1537})
        if 'result' in result and result['result']:
            recognized_text = result['result'][0]
            print('语音识别结果:', recognized_text)
            return recognized_text
        else:
            recognized_text = " "
            print('语音识别失败')
            return recognized_text
        
    def get_file_content(self):
        with open(self.filepath, 'rb') as fp:
            return fp.read()
    
    def post_to_api(self, data):
        """将语音识别结果发送到接口，并接收返回的回复"""
        try:
            print(f"发送数据到接口: {POST_URL}")
            payload = {"message": data}  # 参数格式为 key: "data"  payload = {"data": data}  
            response = requests.post(POST_URL, json=payload)
            if response.status_code == 200:
                return response.json().get("response", "")  # 假设接口返回 {"response": "接口回复内容"}
            else:
                print(f"接口返回错误: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"接口请求错误: {e}")
        return ""
    
    def get_token(self):
    #拼接得到Url
        Url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id="+API_KEY+"&client_secret="+SECRET_KEY
        try:
            resp = urllib_request.urlopen(Url)
            result = json.loads(resp.read().decode('utf-8'))
            # 打印access_token
            print("access_token:",result['access_token'])
            return result['access_token']
        except request.URLError as err:
            print('token http response http code : ' + str(err.code))

    def speech_synthesis(self,text):
        # 1、获取 access_token
        token = self.get_token()
        # 2、将需要合成的文字做2次urlencode编码
        TEXT = text
        tex = parse.quote_plus(TEXT)  # 两次urlencode
        # 3、设置文本以及其他参数
        params = {'tok': token,     # 开放平台获取到的开发者access_token
                'tex': tex,       # 合成的文本，使用UTF-8编码。小于2048个中文字或者英文数字
                'per': 4,         # 发音人选择, 基础音库：0为度小美，1为度小宇，3为度逍遥，4为度丫丫，
                'spd': 5,         # 语速，取值0-15，默认为5中语速
                'pit': 5,         # 音调，取值0-15，默认为5中语调
                'vol': 15,        # 音量，取值0-15，默认为5中音量
                'aue': 6,         # 下载的文件格式, 3为mp3格式(默认); 4为pcm-16k; 5为pcm-8k; 6为wav（内容同pcm-16k）
                'cuid': "7749py", # 用户唯一标识
                'lan': 'zh', 'ctp': 1}  # lan ctp 固定参数
        # 4、将参数编码，然后放入body，生成Request对象
        data = parse.urlencode(params)
        req = urllib_request.Request("http://tsn.baidu.com/text2audio", data.encode('utf-8'))
        # 5、发送post请求
        f = urllib_request.urlopen(req)
        result_str = f.read()
        # 6、将返回的header信息取出并生成一个字典
        headers = dict((name.lower(), value) for name, value in f.headers.items())
        # 7、如果返回的header里有”Content-Type: audio/wav“信息，则合成成功
        if "audio/wav" in headers['content-type'] :
            print("tts success")
            # 合成成功即将数据存入文件
            with open("/home/orangepi/python/gui/result.wav", 'wb') as of:
                of.write(result_str)
    
    def stop_audio(self):
        if self.pygame_initialized:
            pygame.mixer.stop()
            self.pygame_initialized = False
    
    def play_audio(self, filename):
        threading.Thread(target=self._play_audio_thread, args=(filename,)).start()

    def _play_audio_thread(self, filename):
        pygame.init()
        self.pygame_initialized = True
        # 获取输出设备索引
        # output_device_index = self.get_device_index_by_name("USB Audio Device", input=False)
        sound = pygame.mixer.Sound(filename)
        sound.play()
    
    def conversation(self):
        print("进入语音交互...")
        user_input = self.get_audio_and_recognize()
        if not user_input.strip():
            self.speech_synthesis("我没听清，请再说一次！")
            self.play_audio("/home/orangepi/python/gui/result.wav")
            return
        
        # 1. 发送语音识别结果到接口
        response = self.post_to_api(user_input)
        if response:
            print(f"收到接口回复: {response}")
            self.speech_synthesis(response)
        else:
            print("未收到接口回复或请求失败")
            # self.speech_synthesis("暂时无法获取回复，请稍后再试。")
        
        # 2. 播放语音合成结果
        self.play_audio("/home/orangepi/python/gui/result.wav")
 
##############与C++（巡逻设置代码）通信##################
class CoordinateReceiver(QThread):
    update_signal = pyqtSignal(str)  # 定义一个信号，用于在接收到数据时更新界面
    
    def __init__(self):
        super().__init__()
        self.coordinates = []  # 存储坐标数据的列表
        self.running = True  # 控制线程运行的标志
        self.sock = None  # 用于存储socket连接

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', 9998))
            data = sock.recv(1024)
            sock.close()  # 确保关闭连接
            decoded_data = data.decode('utf-8', errors='ignore')
            coordinates = decoded_data.split('\x00')[0]
            print(coordinates)
            self.update_signal.emit(coordinates)  # 发射信号，将接收到的数据发送到主线程
        
        except ConnectionRefusedError:
            # print("连接被拒绝，请确认服务器已启动。")
            self.update_signal.emit("连接失败：服务器未启动")  # 可以发出失败的信号，通知主线程
        except Exception as e:
            print(f"发生错误: {e}")
            self.update_signal.emit(f"发生错误: {e}")  # 将错误信息发射到主线程
        finally:
            if self.sock:
                self.sock.close()  # 确保关闭连接
            # print("C++通信线程结束")
    
    def stop(self):
        """停止线程"""
        self.running = False  # 设置标志为 False，退出循环
        self.quit()  # 停止线程的事件循环

####################机器人表情界面################################
class MainWindow_robot(QMainWindow,Ui_MainWindow_robot):
    # 定义一个信号，用来通知主窗口显示建图页面
    show_build_map_signal = pyqtSignal()
    start_build_signal = pyqtSignal()
    save_map_signal = pyqtSignal()
    update_map_signal = pyqtSignal()

    def __init__(self):  
        super().__init__()  
        global recognition_thread  # 声明全局变量
        self.ui_robot = Ui_MainWindow_robot()  
        self.ui_robot.setupUi(self)  
         # 使用样式表设置背景色为黑色  
        # 为主窗口的特定控件设置样式，避免影响子窗口
        self.setStyleSheet("QWidget { background-color: black; }")  # 只设置中央控件的背景色为黑色 
        self.resize_to_screen()
        self.init_movie() 
        # 初始化子窗口
        self.build_map_window = SubWindow_map()
        # 连接信号到显示建图页面的方法
        self.show_build_map_signal.connect(self.show_build_map)
        self.start_build_signal.connect(self.forward_start_build_signal)
        self.save_map_signal.connect(self.forward_save_map_signal)
        self.update_map_signal.connect(self.forward_update_map_signal)
        # 设置快捷键 Ctrl+Q 关闭窗口
        self.setWindowFlags(Qt.FramelessWindowHint) #隐藏标题栏
        quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        quit_shortcut.activated.connect(self.close) 
         # 启用触摸事件和语音线程
        self.recognition_thread = VoiceRecognitionThread(filepath='/home/orangepi/python/gui/test.wav')
        recognition_thread = self.recognition_thread  # 赋值给全局变量
        self.recognition_thread.start()
        #初始化json文件
        self.saved_data_init()
        self.button_state_init()
         # 添加一个白色圆形按钮
        self.center_button = QPushButton("", self)
        self.center_button.clicked.connect(self.open_web_page)
        button_size = 100  # 按钮的大小
        self.center_button.setFixedSize(button_size, button_size)
        self.center_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                border-radius: 50px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        center_x = (self.width() - button_size) // 2
        center_y = (self.height() - button_size) // 2
        self.center_button.move(center_x, center_y) 

    def init_movie(self):  
        self.movie = QMovie("/home/orangepi/Qt/gui_qt/img/robot.gif")  
        self.ui_robot.label.setMovie(self.movie)  
        self.movie.start()
    
    def show_build_map(self):
        """显示建图页面"""
        self.build_map_window.show()
    
    def forward_start_build_signal(self):
        """将开始建图信号转发给子窗口"""
        self.build_map_window.start_build_signal.emit()
    
    def forward_save_map_signal(self):
        self.build_map_window.save_map_signal.emit()
    
    def forward_update_map_signal(self):
        self.build_map_window.update_map_signal.emit()
  
    def resize_to_screen(self):  
        screen_rect = QDesktopWidget().availableGeometry()  # 获取可用的屏幕几何
        self.setGeometry(screen_rect)  # 设置窗口尺寸为可用屏幕尺寸
        self.showMaximized()  # 窗口最大化
    
    def closeEvent(self, event):
        # 停止线程并关闭窗口
        self.recognition_thread.quit()
        self.recognition_thread.wait()
        super().closeEvent(event)
    
    def saved_data_init(self):
        """读取 saved_data.json 并只初始化 control 字段"""
        file_path = "/home/orangepi/python/gui/saved_data_huake.json"
        default_control_value = 0

        # 如果文件不存在，创建文件并初始化所有字段
        if not os.path.exists(file_path):
            default_data = {
                "control": default_control_value,
            }
            try:
                with open(file_path, "w", encoding="utf-8") as json_file:
                    json.dump(default_data, json_file, ensure_ascii=False, indent=4)
                print(f"初始化 {file_path} 文件成功")
            except IOError as e:
                print(f"初始化文件时发生错误: {e}")
            return

        # 如果文件存在，读取并更新 control 字段
        try:
            with open(file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

            # 仅更新 control 字段
            data["control"] = default_control_value

            # 将更新后的数据重新保存到文件
            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
            print(f"成功更新 {file_path} 中的 control 字段")
        except (IOError, json.JSONDecodeError) as e:
            print(f"读取或更新数据时发生错误: {e}")
    
    def button_state_init(self):
        """初始化 button_state.json 文件内容"""
        default_button_state = {
            "move": True
        }
        self.button_state_path="/home/orangepi/python/gui/button_state.json"

        # 如果文件不存在，创建文件并初始化字段
        if not os.path.exists(self.button_state_path):
            try:
                with open(self.button_state_path, "w", encoding="utf-8") as json_file:
                    json.dump(default_button_state, json_file, ensure_ascii=False, indent=4)
                print(f"初始化 {self.button_state_path} 文件成功")
            except IOError as e:
                print(f"初始化文件时发生错误: {e}")
            return

        # 如果文件存在，检查并更新字段
        try:
            with open(self.button_state_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

            # 更新 human 和 move 字段
            data["move"] = default_button_state["move"]

            # 将更新后的数据重新保存到文件
            with open(self.button_state_path, "w", encoding="utf-8") as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
            print(f"成功更新 {self.button_state_path} 中的字段")
        except (IOError, json.JSONDecodeError) as e:
            print(f"读取或更新数据时发生错误: {e}")
    def open_web_page(self):
        import webbrowser
        url = "http://www.baidu.com"
        webbrowser.open(url)
# Flask 初始化
app = Flask(__name__)
# # 初始按钮状态
# button_state = {
#     "move": True,   # toggle_button_2 对应 "move"
# }

# # 读取按钮状态
# def load_button_state():
#     """加载按钮状态"""
#     global button_state
#     if os.path.exists("button_state_huake.json"):
#         with open("button_state_huake.json", "r") as file:
#             button_state = json.load(file)

# # 保存按钮状态
# def save_button_state():
#     """保存按钮状态到文件"""
#     with open("button_state_huake.json", "w") as file:
#         json.dump(button_state, file)

# @app.route('/get_button_state', methods=['GET'])
# def get_button_state():
#     """返回当前按钮的状态"""
#     try:
#         load_button_state()  # 加载最新状态
#         return jsonify({
#             "status": "success",
#             "code": 200,
#             "data": button_state
#         }), 200
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "code": 500,
#             "message": f"加载状态时发生错误: {str(e)}"
#         }), 500

# # 设置按钮状态
# @app.route('/set_button_state', methods=['POST'])
# def set_button_state():
#     """设置按钮的状态"""
#     global button_state
#     try:
#         data = request.get_json()  # 获取 JSON 请求体

#         if not data or 'move' not in data:
#             return jsonify({
#                             "status": "error",
#                             "code": 400,
#                             "message": "缺少必需的字段: 'move'"
#                         }), 400

#         # 更新按钮状态
#         button_state["move"] = data["move"]
#         save_button_state()  # 保存状态到文件

#         with open("saved_data_huake.json", "r", encoding="utf-8") as file:
#             saved_data = json.load(file)

#         # 修改 control 字段，根据 move 按钮的状态设置为 1 或 0
#         saved_data["control"] = 0 if button_state["move"] else 1

#         # 保存修改后的 saved_data.json
#         with open("saved_data_huake.json", "w", encoding="utf-8") as file:
#             json.dump(saved_data, file, ensure_ascii=False, indent=4)

#         # 生成更详细的 message 字段
#         move_status = "打开" if button_state["move"] else "关闭"

#         return jsonify({
#             "status": "success",
#             "code": 200,
#             "message": f"人脸识别: {human_status}, 定位模式: {move_status}"
#         }), 200

#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "code": 500,
#             "message": f"设置按钮状态时发生错误: {str(e)}"
#         }), 500


# 1. 开始建图
@app.route('/startbuildmap', methods=['POST'])
def start_build_map_url():
    """启动建图功能"""
    try:
        main_window.start_build_signal.emit()
        threading.Thread(target=trigger_signal_show_build_map).start()
        return jsonify({"code": 200, "status": "success", "message": "建图已开始"})
    except Exception as e:
        return jsonify({"code": 500, "status": "error", "message": str(e)}), 500

def trigger_signal_show_build_map():
    """在 Flask 后端触发信号来显示建图页面"""
    main_window.show_build_map_signal.emit()

# 2. 保存地图到本地
@app.route('/savemap', methods=['POST'])
def save_local_map_url():
    """保存地图到本地"""
    try:
        main_window.save_map_signal.emit()
        return jsonify({"code": 200, "status": "success", "message": f"地图已保存到"})
    except Exception as e:
        return jsonify({"code": 500, "status": "error", "message": str(e)}), 500

# 3. 更新机器人地图
@app.route('/updatemap', methods=['POST'])
def update_robot_map_url():
    """上传最新地图到机器人"""
    try:
        main_window.update_map_signal.emit()
        return jsonify({"code": 200, "status": "success", "message": f"地图已上传到机器人"})
    except Exception as e:
        return jsonify({"code": 500, "status": "error", "message": str(e)}), 500

# 设置 USB 音频设备音量的函数
def set_usb_audio_volume(volume_level):
    device = "alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.stereo-fallback"
    try:
        # 将设备设置为默认输出设备
        subprocess.run(['pactl', 'set-default-sink', device], check=True)
        # 调用 pactl 设置音量
        subprocess.run(['pactl', 'set-sink-volume', device, f'{volume_level}%'], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

# 4.音量控制接口
@app.route('/setvolume', methods=['POST'])
def set_volume():
    # 从请求中获取音量参数
    data = request.json
    if 'vol' not in data:
        return jsonify({"code": 400,"error": "Missing 'vol' parameter"}), 400

    volume = data['vol']
    
    # 检查音量值是否在有效范围内
    if not (0 <= volume <= 153):
        return jsonify({"code": 400,"error": "Volume must be between 0 and 100"}), 400

    # 设置音量
    if set_usb_audio_volume(volume):
        return jsonify({"code": 200,"success": f"Volume set to {volume}%"}), 200
    else:
        return jsonify({"code": 500,"error": "Failed to set volume"}), 500

# 5.语音播放接口
@app.route('/robotsay', methods=['POST'])
def say():
    data = request.get_json()
    if not data or 'txt' not in data:
        return jsonify({"code": 400, "error": "请求参数不正确"}), 400

    text = data['txt']
    if not text.strip():
        return jsonify({"code": 400, "error": "文本不能为空"}), 400

    try:
        # 调用语音合成和播放方法
        recognition_thread.speech_synthesis(text)
        recognition_thread.play_audio("/home/orangepi/python/gui/result.wav")
        return jsonify({"code": 200, "message": "语音播放成功"}), 200
    except Exception as e:
        return jsonify({"code": 500, "error": f"语音播放失败: {str(e)}"}), 500

# 启动 Flask 服务器
def run_flask_server():
    app.run(host="0.0.0.0", port=5000, threaded=True)  # 启动 Flask 服务器，监听所有 IP

# 启动 Flask 服务器的线程
flask_thread = threading.Thread(target=run_flask_server)
flask_thread.daemon = True
flask_thread.start()

##############建图功能界面##################
class SubWindow_map(QMainWindow, Ui_SubWindow_map):
    # 定义一个自定义信号
    start_build_signal = pyqtSignal()
    save_map_signal = pyqtSignal()
    update_map_signal = pyqtSignal()

    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui_map = Ui_SubWindow_map()
        self.ui_map.setupUi(self)
        self.setWindowTitle("建图")
        self.setStyleSheet("background-color: #efefef;") 

        # 将窗口移动到屏幕中央
        self.center()

        # 初始化 QGraphicsScene 用于显示地图
        self.scene = QGraphicsScene(self)
        self.ui_map.graphicsView.setScene(self.scene)

        self.ui_map.pushButton.clicked.connect(self.start_build_map)
        self.ui_map.pushButton_2.clicked.connect(self.save_local_map)
        self.ui_map.pushButton_3.clicked.connect(self.update_robot_map)

        # 定时器用于刷新地图数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_map_data)

        # 连接信号与槽
        self.start_build_signal.connect(self.start_build_map)
        self.save_map_signal.connect(self.save_local_map)
        self.update_map_signal.connect(self.update_robot_map)

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)  # 设置为错误图标
        msg_box.setWindowTitle("错误")
        msg_box.setText(message)
        msg_box.exec_()

    def show_success_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)  # 设置为信息图标 
        msg_box.setWindowTitle("成功")
        msg_box.setText(message)
        msg_box.exec_()

    def start_build_map(self):
        url_map = "http://192.168.31.221:1448/api/core/slam/v1/maps"
        url_power = "http://192.168.31.221:1448/api/core/system/v1/power/status" 

        try:
            response = requests.get(url_power)
            
            data = response.json()
            print("完整的响应数据：", data)

            # 提取特定键的值
            key = "dockingStatus"
            if key in data:

                # print(f"特定键 '{key}' 的值：", data[key])
                if data[key] == "on_dock":

                    response = requests.delete(url_map)

                elif data[key] == "not_on_dock":
                     self.show_error_message("机器人不再桩上")
                     return

            else:
                print(f"响应中未找到特定键 '{key}'")

        except Exception as e:
            print(f"Error occurred: {e}")  

        # 开启定时器，模拟实时获取数据
        self.timer.start(500)  # 每 1 秒刷新一次数据
    
    def update_map_data(self):

        url = "http://192.168.31.221:1448/api/core/slam/v1/maps/explore"
        try:
            # 向服务端发送 GET 请求
            response = requests.get(url)
            
            # 确保响应是成功的
            if response.status_code == 200:
                data = response.content
                data_length = len(data)
                print(f"Received {data_length} bytes from server.")
                
                if data_length >= 36:
                    # 解析前36字节的元数据,'<2f2I1f12sI'为解包格式
                    x_start, y_start, x_grids, y_grids, resolution, reserved, total_bytes = struct.unpack('<2f2I1f12sI', data[:36])
                    
                    print(f"Metadata - X Start: {x_start}, Y Start: {y_start}")
                    print(f"X Grids: {x_grids}, Y Grids: {y_grids}, Resolution: {resolution}")
                    print(f"Total bytes for map data: {total_bytes}")
                    
                    # 检查后续的地图数据字节数是否匹配
                    if len(data[36:]) == total_bytes:
                        map_data = data[36:]
                        
                        # 将元数据和地图数据保存到文本文件
                        # save_data_to_file("map_data.txt", x_start, y_start, x_grids, y_grids, resolution, map_data)

                        self.generate_map_image(x_grids,y_grids,map_data)

                    else:
                        print("Error: Map data size does not match the expected size.")
                else:
                    print("Error: Received data is too short to contain valid map data.")
            else:
                print(f"Failed to fetch data. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error occurred: {e}")

    def generate_map_image(self, width, height, map_data):

        try:

            # 创建一个QImage对象，大小为width * height，格式为RGB32
            map_image = QImage(width, height, QImage.Format_RGB32)

            # 遍历地图数据并设置像素点颜色
            for i in range(len(map_data)):
                x = i % width
                y = int(i / width)

                # 配置该点的颜色
                if map_data[i] > 127:
                    color = QColor(Qt.black)
                elif map_data[i] == 0:
                    color = QColor(Qt.gray)
                else:
                    color = QColor(Qt.white)
                    
                # 设置像素颜色
                map_image.setPixel(x, y, color.rgb())

            
            map_image.save("/home/orangepi/swy/qt_ws/map.png")
            pixmap = QPixmap.fromImage(map_image)
            self.scene.clear()  # 清除上次显示的图像
            self.scene.addItem(QGraphicsPixmapItem(pixmap))

            print("Map imag updated successfully")


        except Exception as e:
            print(f"Error occurred while writing to file: {e}")

    def save_local_map(self):
        
        url = "http://192.168.31.221:1448/api/core/slam/v1/maps/stcm"
        file_path = "/home/orangepi/swy/qt_ws/map.stcm"
        
        try:

            response = requests.get(url)
            if response.status_code == 200:
                data = response.content
                data_length = len(data)
                print(f"Received {data_length} bytes from server.")

                # 写入数据(以二进制模式写入
                with open(file_path,'wb') as file:
                    file.write(data)
                print(f"Map data saved to {file_path}")
                self.show_success_message("当前地图已保存")
            
            else:
                print(f"Failed to fetch data. Status code: {response.status_code}")

        except Exception as e:
            print(f"Error occurred: {e}")

    def update_robot_map(self):

        url = "http://192.168.31.221:1448/api/core/slam/v1/maps/stcm"
        file_path = "/home/orangepi/swy/qt_ws/map.stcm"

        url_power = "http://192.168.31.221:1448/api/core/system/v1/power/status" 

        try:
            response = requests.get(url_power)
            
            data = response.json()
            print("完整的响应数据：", data)

            # 提取特定键的值
            key = "dockingStatus"
            if key in data:

                # print(f"特定键 '{key}' 的值：", data[key])
                if data[key] == "on_dock":
                    with open(file_path,'rb') as file:
                        response = requests.put(url,data=file)
                    
                    if response.status_code in [200,201]:
                        print(f"File '{file_path}' successfully uploaded to {url}")
                        self.show_success_message("当前地图已更新至固件")
                        self.close()  # 关闭窗口
                    else:
                        print(f"Failed to fetch data. Status code: {response.status_code}")
                    
                    
                elif data[key] == "not_on_dock":
                     self.show_error_message("请回桩后再更新")
                     return

            else:
                print(f"响应中未找到特定键 '{key}'")

        except Exception as e:
            print(f"Error occurred: {e}")  
    
    def center(self):
        # 获取屏幕大小和窗口大小
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def closeEvent(self, event):
        event.accept()
        self.deleteLater()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow_robot()
    main_window.show()
    # 启用全屏模式
    # main_window.showFullScreen()
    sys.exit(app.exec_())