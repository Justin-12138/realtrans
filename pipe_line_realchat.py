"""
实时英语对话助手 v1.0

流程:
1. 语音录制 (VAD 自适应检测)
2. ASR 识别 (Whisper - 中文→文本)
3. LLM 对话 (OpenAI 兼容 API - 英语伙伴助手)
4. TTS 播放 (Kyutai TTS - 文本→语音)

特性:
- 无状态对话 (每轮独立，不携带历史)
- 英语学习伙伴风格
- 自适应噪音校准
- 播放期间禁用麦克风 (避免回声)

依赖: pip install faster-whisper requests pyaudio webrtcvad numpy openai
"""

import pyaudio
import wave
import io
import requests
import webrtcvad
import collections
import time
import json
import numpy as np
import threading
from enum import Enum
from faster_whisper import WhisperModel
from typing import Optional
from dataclasses import dataclass

# ============== 配置 ==============
LLM_API_URL = "http://127.0.0.1:6001/v1/chat/completions"
TTS_API_URL = "http://127.0.0.1:9099/tts"

# 音频参数
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# VAD 配置
VAD_MODE = 2
SPEECH_START_FRAMES = 10
SPEECH_END_SILENCE_MS = 600
SPEECH_MIN_DURATION = 0.8
SPEECH_MAX_DURATION = 20

# 自适应音量阈值
VOLUME_THRESHOLD_INIT = 300
VOLUME_THRESHOLD_MIN = 150
VOLUME_THRESHOLD_MAX = 800
CALIBRATION_FRAMES = 50

# 重试配置
MAX_RETRIES = 2
TIMEOUT_LLM = 8
TIMEOUT_TTS = 15

# LLM 配置
LLM_MODEL = "Qwen3-0.6B"
SYSTEM_PROMPT = (
    "You are a friendly English conversation partner helper. "
    "The user is learning English and will speak in Chinese. "
    "Respond naturally in English to help them practice. "
    "Keep responses concise (1-2 sentences) and conversational. "
    "Be encouraging and helpful."
)


class State(Enum):
    IDLE = "空闲"
    CALIBRATING = "校准中"
    LISTENING = "监听中"
    RECORDING = "录音中"
    PROCESSING = "处理中"
    PLAYING = "播放中"


@dataclass
class AudioStats:
    """音频统计信息"""
    rms: float
    max_amplitude: int
    duration: float


class AdaptiveVAD:
    """自适应 VAD 检测器"""

    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.volume_threshold = VOLUME_THRESHOLD_INIT
        self.noise_floor = 0
        self.calibrated = False

    def calibrate(self, stream, frames_count: int = CALIBRATION_FRAMES):
        """校准环境噪音水平"""
        print(f"[校准] 正在采样环境噪音... (保持安静 {frames_count * 30}ms)")

        noise_samples = []
        for i in range(frames_count):
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_np = np.frombuffer(chunk, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_np.astype(float) ** 2))
            noise_samples.append(rms)

            progress = (i + 1) / frames_count * 100
            bar_len = int(progress / 5)
            print(f"\r  进度: [{'█' * bar_len}{'·' * (20 - bar_len)}] {progress:.0f}%", end='')

        print()

        self.noise_floor = np.percentile(noise_samples, 75)
        self.volume_threshold = max(
            VOLUME_THRESHOLD_MIN,
            min(VOLUME_THRESHOLD_MAX, self.noise_floor * 2)
        )
        self.calibrated = True

        print(f"[校准] ✓ 完成 | 噪音基线: {self.noise_floor:.0f} | 阈值: {self.volume_threshold:.0f}")

    def is_speech(self, chunk: bytes) -> tuple[bool, float]:
        """检测是否为语音，返回 (是否语音, RMS音量)"""
        audio_np = np.frombuffer(chunk, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_np.astype(float) ** 2))

        try:
            vad_result = self.vad.is_speech(chunk, SAMPLE_RATE)
        except:
            vad_result = False

        volume_ok = rms > self.volume_threshold

        return vad_result and volume_ok, rms


class EnglishChatPartner:
    """英语对话助手"""

    def __init__(self):
        self.vad = AdaptiveVAD()
        self.audio = pyaudio.PyAudio()
        self.whisper_model = None
        self.state = State.IDLE
        self.is_running = False
        self.stats = {'processed': 0, 'failed': 0, 'avg_time': 0}

    def log(self, msg: str, prefix: str = ""):
        """日志输出"""
        timestamp = time.strftime('%H:%M:%S')
        print(f"[{timestamp}] {prefix}{msg}")

    def set_state(self, new_state: State):
        """切换状态"""
        if self.state != new_state:
            self.state = new_state

    def _bytes_to_wav_bytes(self, audio_data: bytes) -> bytes:
        """Convert raw audio bytes to WAV format in memory"""
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        return wav_buffer.getvalue()

    def _load_whisper_in_background(self):
        """后台加载 Whisper 模型"""
        self.whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
            num_workers=2
        )

    def init_whisper(self, wait=True):
        """加载 Whisper 模型"""
        if wait:
            self.log("正在加载 Whisper 模型 (base)...")
            self._load_whisper_in_background()
            self.log("✓ 模型加载完成", "  ")
        else:
            # 后台加载，不等待
            thread = threading.Thread(target=self._load_whisper_in_background, daemon=True)
            thread.start()
            return thread

    def get_audio_stats(self, audio_data: bytes) -> AudioStats:
        """计算音频统计信息"""
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_np.astype(float) ** 2))
        max_amp = np.max(np.abs(audio_np))
        duration = len(audio_data) / SAMPLE_RATE
        return AudioStats(rms=rms, max_amplitude=max_amp, duration=duration)

    def record_speech(self, stream) -> Optional[bytes]:
        """录音直到检测到语音结束"""
        self.set_state(State.LISTENING)

        # 等待语音开始
        start_buffer = collections.deque(maxlen=SPEECH_START_FRAMES)

        self.log("🎧 等待说话...")
        while self.is_running:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            is_voice, rms = self.vad.is_speech(chunk)
            start_buffer.append((chunk, is_voice, rms))

            voice_ratio = sum(1 for _, v, _ in start_buffer if v) / len(start_buffer)

            if voice_ratio >= 0.85 and len(start_buffer) == SPEECH_START_FRAMES:
                break

        if not self.is_running:
            return None

        # 开始录音
        self.set_state(State.RECORDING)
        self.log("🎤 录音中...", "  ")

        frames = [f for f, _, _ in start_buffer]
        silence_frames = 0
        record_start = time.time()
        max_rms = max(r for _, _, r in start_buffer)

        last_update = time.time()

        while self.is_running:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(chunk)

            is_voice, rms = self.vad.is_speech(chunk)
            duration = time.time() - record_start

            if rms > max_rms:
                max_rms = rms

            if time.time() - last_update > 0.1:
                indicator = "🔊" if is_voice else "  "
                bar_len = min(int(duration * 2), 30)
                bar = "█" * bar_len
                print(f"\r  ⏺️  {duration:4.1f}s {indicator} [{bar:<30}] RMS:{rms:4.0f}", end='', flush=True)
                last_update = time.time()

            # 静音检测
            if is_voice:
                silence_frames = 0
            else:
                silence_frames += 1
                silence_ms = silence_frames * CHUNK_DURATION_MS

                if silence_ms >= SPEECH_END_SILENCE_MS:
                    if rms < max_rms * 0.3:
                        print()
                        self.log(f"✓ 录音结束 (时长: {duration:.1f}s)", "  ")
                        break

            # 超时保护
            if duration >= SPEECH_MAX_DURATION:
                print()
                self.log(f"⏱️  达到最大时长 ({SPEECH_MAX_DURATION}s)", "  ")
                break

        if not self.is_running:
            return None

        # 检查录音质量
        audio_data = b''.join(frames)
        stats = self.get_audio_stats(audio_data)

        if stats.duration < SPEECH_MIN_DURATION:
            self.log(f"⚠️  录音太短 ({stats.duration:.1f}s < {SPEECH_MIN_DURATION}s)，已忽略", "  ")
            return None

        if stats.max_amplitude < 500:
            self.log(f"⚠️  音量过低 (峰值: {stats.max_amplitude})，已忽略", "  ")
            return None

        return audio_data

    def asr_transcribe(self, audio_data: bytes) -> Optional[str]:
        """ASR: 音频 → 文本"""
        wav_bytes = self._bytes_to_wav_bytes(audio_data)
        wav_io = io.BytesIO(wav_bytes)

        try:
            segments, info = self.whisper_model.transcribe(
                wav_io,
                beam_size=5,
                language="zh"
            )
            text = "".join(seg.text for seg in segments).strip()
            return text if text else None
        except Exception as e:
            self.log(f"❌ ASR 错误: {e}", "  ")
            return None

    def chat_with_llm(self, text: str, retries: int = MAX_RETRIES) -> Optional[str]:
        """LLM 对话 (无状态)"""
        for attempt in range(retries):
            try:
                response = requests.post(
                    LLM_API_URL,
                    json={
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": text}
                        ],
                        "max_tokens": 256,
                        "stream": False
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=TIMEOUT_LLM
                )
                response.raise_for_status()
                result = response.json()["choices"][0]["message"]["content"].strip()
                return result if result else None
            except requests.Timeout:
                if attempt < retries - 1:
                    self.log(f"⚠️  LLM 超时，重试 {attempt + 1}/{retries}...", "  ")
                    time.sleep(0.5)
                else:
                    self.log(f"❌ LLM 失败: 超时", "  ")
            except Exception as e:
                if attempt < retries - 1:
                    self.log(f"⚠️  LLM 错误，重试 {attempt + 1}/{retries}...", "  ")
                    time.sleep(0.5)
                else:
                    self.log(f"❌ LLM 失败: {e}", "  ")
        return None

    def chat_with_llm_stream(self, text: str, retries: int = MAX_RETRIES) -> Optional[str]:
        """LLM 对话 (流式，可提前开始TTS)"""
        for attempt in range(retries):
            try:
                response = requests.post(
                    LLM_API_URL,
                    json={
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": text}
                        ],
                        "max_tokens": 256,
                        "stream": True
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=TIMEOUT_LLM,
                    stream=True
                )
                response.raise_for_status()

                full_text = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    full_text += content
                        except:
                            pass

                return full_text.strip() if full_text else None

            except requests.Timeout:
                if attempt < retries - 1:
                    self.log(f"⚠️  LLM 超时，重试 {attempt + 1}/{retries}...", "  ")
                    time.sleep(0.5)
                else:
                    self.log(f"❌ LLM 失败: 超时", "  ")
            except Exception as e:
                if attempt < retries - 1:
                    self.log(f"⚠️  LLM 错误，重试 {attempt + 1}/{retries}...", "  ")
                    time.sleep(0.5)
                else:
                    self.log(f"❌ LLM 失败: {e}", "  ")
        return None

    def process_audio(self, audio_data: bytes) -> Optional[str]:
        """处理音频：ASR + LLM"""
        self.set_state(State.PROCESSING)

        # ASR
        self.log("📝 识别中...", "  ")
        user_text = self.asr_transcribe(audio_data)
        if not user_text:
            self.log("⚠️  未识别到文本", "  ")
            return None
        self.log(f"   用户: {user_text}", "  ")

        # LLM (使用流式，降低首字延迟)
        self.log("🤖 思考中...", "  ")
        response = self.chat_with_llm_stream(user_text)
        if not response:
            return None
        self.log(f"   助手: {response}", "  ")

        return response

    def play_tts(self, text: str, stream):
        """TTS 并播放 (内存缓冲，无磁盘IO)"""
        self.set_state(State.PLAYING)
        self.log("🔊 生成语音...", "  ")

        stream.stop_stream()

        try:
            response = requests.post(
                TTS_API_URL,
                data={"text": text},
                timeout=TIMEOUT_TTS
            )
            response.raise_for_status()

            # 直接在内存中处理 WAV
            wav_io = io.BytesIO(response.content)

            with wave.open(wav_io, 'rb') as wf:
                play_stream = self.audio.open(
                    format=self.audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )

            self.log("🔊 播放中...", "  ")

            # 从内存流式播放
            wav_io.seek(44)  # 跳过 WAV header
            while self.is_running:
                data = wav_io.read(1024)
                if not data:
                    break
                play_stream.write(data)

            play_stream.stop_stream()
            play_stream.close()

            self.log("✓ 播放完成", "  ")

        except Exception as e:
            self.log(f"❌ TTS/播放错误: {e}", "  ")
        finally:
            time.sleep(0.5)
            stream.start_stream()

    def run_cycle(self, stream) -> bool:
        """执行一次完整循环"""
        start_time = time.time()

        # 1. 录音
        audio_data = self.record_speech(stream)
        if audio_data is None:
            return self.is_running

        # 2. 处理
        response_text = self.process_audio(audio_data)
        if response_text is None:
            self.stats['failed'] += 1
            return self.is_running

        # 3. 播放
        self.play_tts(response_text, stream)

        # 统计
        elapsed = time.time() - start_time
        self.stats['processed'] += 1
        self.stats['avg_time'] = (
            (self.stats['avg_time'] * (self.stats['processed'] - 1) + elapsed)
            / self.stats['processed']
        )

        self.log(f"✓ 本轮完成 (耗时: {elapsed:.1f}s)", "  ")
        print()

        return self.is_running

    def start(self):
        """启动系统"""
        self.is_running = True

        print(f"\n{'='*60}")
        print("🗣️  实时英语对话助手 v1.0 (优化版)")
        print(f"{'='*60}\n")

        # 并行加载: 启动模型加载线程
        print("📦 正在初始化...")
        model_thread = self.init_whisper(wait=False)

        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
        except Exception as e:
            self.log(f"❌ 无法打开麦克风: {e}")
            return

        # 校准期间模型在后台加载
        self.vad.calibrate(stream)

        # 等待模型加载完成
        if model_thread:
            self.log("⏳ 等待模型加载...", "  ")
            model_thread.join()
            self.log("✓ 模型就绪", "  ")

        print(f"\n{'='*60}")
        print("📋 系统就绪")
        print(f"   流程: 监听 → 录音 → 识别 → 对话 → 播放")
        print(f"   模式: 无状态对话 (每轮独立)")
        print(f"   操作: 说中文练习英语 | Ctrl+C 退出")
        print(f"{'='*60}\n")

        try:
            while self.is_running:
                if not self.run_cycle(stream):
                    break
        except KeyboardInterrupt:
            print("\n")
            self.log("用户中断")
        finally:
            self.is_running = False
            stream.stop_stream()
            stream.close()
            self.audio.terminate()

            print(f"\n{'='*60}")
            print("📊 运行统计")
            print(f"   成功: {self.stats['processed']} | 失败: {self.stats['failed']}")
            if self.stats['processed'] > 0:
                print(f"   平均耗时: {self.stats['avg_time']:.1f}s")
            print(f"{'='*60}")
            self.log("系统已退出 👋")


if __name__ == "__main__":
    partner = EnglishChatPartner()
    partner.start()
