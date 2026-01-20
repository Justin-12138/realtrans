# RealTrans - 实时语音翻译系统

一个实时语音翻译系统，可将中文语音转写、翻译成英文，并合成语音输出。

## 特性

- **自适应 VAD (语音活动检测)**: 根据不同环境动态调整音量阈值
- **智能静音检测**: 基于能量衰减曲线，实现准确的语音分割
- **内存音频处理**: 减少文件 I/O，提升性能
- **自动重试**: 优雅处理网络错误
- **播放隔离**: TTS 播放期间禁用麦克风，防止回声
- **进度显示**: 录音和处理过程中实时反馈

## 处理流程

```
麦克风 -> ASR (Whisper) -> 翻译 API -> TTS -> 扬声器
   |          |               |            |
   v          v               v            v
 音频      中文文本        英文文本      音频
```

## 环境要求

- Python >= 3.12
- Linux (PyAudio 依赖 PortAudio)

## 安装

```bash
# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -r requirements.txt
```

## 依赖项

| 包名 | 用途 |
|------|------|
| faster-whisper | 语音识别 (Whisper 模型) |
| pyaudio | 音频输入/输出 |
| requests | HTTP API 客户端 |
| webrtcvad | 语音活动检测 |
| numpy | 音频信号处理 |
| scipy | 音频文件读写 |
| torch | 机器学习模型后端 |
| transformers | 翻译模型 |
| pocket-tts | 文本转语音 (本地可编辑包) |

## 外部服务

系统需要两个外部服务运行：

### 1. 翻译服务 (HY-MT1.5-1.8B)

```bash
# 默认端点: http://127.0.0.1:8099/v1/chat/completions
# OpenAI 兼容格式
```

### 2. TTS 服务 (Pocket-TTS)

```bash
# 启动 pocket-tts 服务器
cd 3parts/pocket-tts
pocket-tts serve --port 9099

# 或使用 API 端点: http://127.0.0.1:9099/tts
```

## 使用方法

### 主程序

```bash
python main.py
```

**工作流程：**
1. 系统校准环境噪音（保持安静约 1.5 秒）
2. 等待 "等待说话..." 提示
3. 说中文
4. 系统转写、翻译并播放英文语音
5. 按 `Ctrl+C` 退出

### 测试脚本

```bash
# 仅测试 ASR
python tests/test_aasr.py

# 仅测试翻译模型
python tests/test_translate.py

# 仅测试 TTS
python tests/test_pockettts.py

# 测试完整流程 (使用音频文件)
python tests/pipeline_all.py
```

## 配置

编辑 `main.py` 进行自定义：

```python
# 音频参数
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30

# VAD 设置
VAD_MODE = 2
SPEECH_START_FRAMES = 10       # 连续 300ms 触发录音
SPEECH_END_SILENCE_MS = 600    # 静音 600ms 停止录音
SPEECH_MIN_DURATION = 0.8      # 最短录音时长 (秒)
SPEECH_MAX_DURATION = 20       # 最长录音时长 (秒)

# API 端点
TRANSLATION_API_URL = "http://127.0.0.1:8099/v1/chat/completions"
TTS_API_URL = "http://127.0.0.1:9099/tts"

# 重试设置
MAX_RETRIES = 2
TIMEOUT_TRANSLATION = 8
TIMEOUT_TTS = 15
```

## 项目结构

```
realtrans/
├── main.py                  # 主程序
├── pyproject.toml           # 项目配置
├── tests/
│   ├── test_aasr.py         # ASR 测试
│   ├── test_translate.py    # 翻译测试
│   ├── test_pockettts.py    # TTS 测试
│   ├── pipeline_all.py      # 完整流程测试
│   └── pipeline_realtime.py # main.py 副本
├── models/
│   ├── curl_translate.sh    # 翻译 API 测试
│   └── tts_api.json         # TTS API 规范
└── 3parts/
    └── pocket-tts/          # 本地 TTS 包 (Kyutai)
```

## 许可证

MIT
