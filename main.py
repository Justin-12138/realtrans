"""
实时语音翻译系统 v4.0 - 深度优化版

核心改进:
1. 自适应 VAD：动态调整音量阈值（适应不同环境）
2. 智能静音检测：基于能量衰减曲线
3. 内存音频处理：减少文件 I/O
4. 重试机制：网络错误自动重试
5. 播放隔离：播放期间禁用麦克风
6. 完善的进度显示

依赖: pip install faster-whisper requests pyaudio webrtcvad numpy
"""

import pyaudio
import wave
import io
import requests
import webrtcvad
import collections
import time
import numpy as np
from enum import Enum
from faster_whisper import WhisperModel
from typing import Optional
from dataclasses import dataclass

# ============== 配置 ==============
TRANSLATION_API_URL = "http://127.0.0.1:8099/v1/chat/completions"
TTS_API_URL = "http://127.0.0.1:9099/tts"

# 音频参数
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# VAD 配置（优化后）
VAD_MODE = 2
SPEECH_START_FRAMES = 10       # 连续10帧才触发（300ms）
SPEECH_END_SILENCE_MS = 600    # 静音600ms结束（更快响应）
SPEECH_MIN_DURATION = 0.8      # 最短录音（秒）
SPEECH_MAX_DURATION = 20       # 最长录音（秒）

# 自适应音量阈值
VOLUME_THRESHOLD_INIT = 300    # 初始阈值
VOLUME_THRESHOLD_MIN = 150     # 最小阈值
VOLUME_THRESHOLD_MAX = 800     # 最大阈值
CALIBRATION_FRAMES = 50        # 校准帧数

# 重试配置
MAX_RETRIES = 2
TIMEOUT_TRANSLATION = 8
TIMEOUT_TTS = 15


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
            
            # 进度条
            progress = (i + 1) / frames_count * 100
            bar_len = int(progress / 5)
            print(f"\r  进度: [{'█' * bar_len}{'·' * (20 - bar_len)}] {progress:.0f}%", end='')
        
        print()
        
        # 计算噪音基线（取75分位数，排除异常值）
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
        
        # VAD 检测
        try:
            vad_result = self.vad.is_speech(chunk, SAMPLE_RATE)
        except:
            vad_result = False
        
        # 音量检测（相对于噪音基线）
        volume_ok = rms > self.volume_threshold
        
        return vad_result and volume_ok, rms


class RealtimeTranslator:
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
            # 不打印每次状态切换，避免刷屏
    
    def init_whisper(self):
        """加载 Whisper 模型"""
        self.log("正在加载 Whisper 模型 (base)...")
        self.whisper_model = WhisperModel(
            "base", 
            device="cpu", 
            compute_type="int8",
            num_workers=2
        )
        self.log("✓ 模型加载完成", "  ")
    
    def get_audio_stats(self, audio_data: bytes) -> AudioStats:
        """计算音频统计信息"""
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_np.astype(float) ** 2))
        max_amp = np.max(np.abs(audio_np))
        duration = len(audio_np) / SAMPLE_RATE
        return AudioStats(rms=rms, max_amplitude=max_amp, duration=duration)
    
    def record_speech(self, stream) -> Optional[bytes]:
        """
        录音直到检测到语音结束
        使用滑动窗口 + 能量衰减检测
        """
        self.set_state(State.LISTENING)
        
        # 等待语音开始
        start_buffer = collections.deque(maxlen=SPEECH_START_FRAMES)
        
        self.log("🎧 等待说话...")
        while self.is_running:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            is_voice, rms = self.vad.is_speech(chunk)
            start_buffer.append((chunk, is_voice, rms))
            
            # 计算语音帧比例
            voice_ratio = sum(1 for _, v, _ in start_buffer if v) / len(start_buffer)
            
            # 85% 以上的帧是语音才触发（比之前更严格）
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
        max_rms = max(r for _, _, r in start_buffer)  # 记录峰值音量
        
        # 实时进度显示（单行更新）
        last_update = time.time()
        
        while self.is_running:
            chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(chunk)
            
            is_voice, rms = self.vad.is_speech(chunk)
            duration = time.time() - record_start
            
            # 更新峰值音量
            if rms > max_rms:
                max_rms = rms
            
            # 进度显示（100ms 更新一次）
            if time.time() - last_update > 0.1:
                indicator = "🔊" if is_voice else "  "
                bar_len = min(int(duration * 2), 30)  # 最多30个字符
                bar = "█" * bar_len
                print(f"\r  ⏺️  {duration:4.1f}s {indicator} [{bar:<30}] RMS:{rms:4.0f}", end='', flush=True)
                last_update = time.time()
            
            # 静音检测（改进版）
            if is_voice:
                silence_frames = 0
            else:
                silence_frames += 1
                silence_ms = silence_frames * CHUNK_DURATION_MS
                
                # 静音达到阈值 + 音量明显下降
                if silence_ms >= SPEECH_END_SILENCE_MS:
                    # 检查音量是否衰减到峰值的30%以下
                    if rms < max_rms * 0.3:
                        print()  # 换行
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
        """ASR: 内存音频 → 文本（避免文件 I/O）"""
        # 创建内存 WAV 文件
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        
        # 保存到临时文件（Faster-Whisper 暂不支持 BytesIO）
        temp_file = "/tmp/realtime_audio.wav"
        with open(temp_file, 'wb') as f:
            f.write(wav_buffer.getvalue())
        
        try:
            segments, info = self.whisper_model.transcribe(
                temp_file,
                beam_size=5,
                language="zh",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300)
            )
            text = "".join(seg.text for seg in segments).strip()
            return text if text else None
        except Exception as e:
            self.log(f"❌ ASR 错误: {e}", "  ")
            return None
    
    def translate_with_retry(self, text: str, retries: int = MAX_RETRIES) -> Optional[str]:
        """翻译（带重试）"""
        for attempt in range(retries):
            try:
                response = requests.post(
                    TRANSLATION_API_URL,
                    json={
                        "model": "HY-MT1.5-1.8B",
                        "messages": [{
                            "role": "user",
                            "content": f"Translate the following segment into English, without additional explanation.\n\n{text}"
                        }],
                        "stream": False
                    },
                    headers={"Authorization": "Bearer sk-1234", "Content-Type": "application/json"},
                    timeout=TIMEOUT_TRANSLATION
                )
                response.raise_for_status()
                result = response.json()["choices"][0]["message"]["content"].strip()
                return result if result else None
            except requests.Timeout:
                if attempt < retries - 1:
                    self.log(f"⚠️  翻译超时，重试 {attempt + 1}/{retries}...", "  ")
                    time.sleep(0.5)
                else:
                    self.log(f"❌ 翻译失败: 超时", "  ")
            except Exception as e:
                if attempt < retries - 1:
                    self.log(f"⚠️  翻译错误，重试 {attempt + 1}/{retries}...", "  ")
                    time.sleep(0.5)
                else:
                    self.log(f"❌ 翻译失败: {e}", "  ")
        return None
    
    def process_audio(self, audio_data: bytes) -> Optional[str]:
        """处理音频：ASR + 翻译"""
        self.set_state(State.PROCESSING)
        
        # ASR
        self.log("📝 识别中...", "  ")
        chinese_text = self.asr_transcribe(audio_data)
        if not chinese_text:
            self.log("⚠️  未识别到文本", "  ")
            return None
        self.log(f"   中文: {chinese_text}", "  ")
        
        # 翻译
        self.log("🌍 翻译中...", "  ")
        english_text = self.translate_with_retry(chinese_text)
        if not english_text:
            return None
        self.log(f"   英文: {english_text}", "  ")
        
        return english_text
    
    def play_tts(self, text: str, stream):
        """TTS 并播放（播放期间暂停录音）"""
        self.set_state(State.PLAYING)
        self.log("🔊 生成语音...", "  ")
        
        # 暂停麦克风
        stream.stop_stream()
        
        try:
            # TTS
            response = requests.post(
                TTS_API_URL,
                data={"text": text},
                stream=True,
                timeout=TIMEOUT_TTS
            )
            response.raise_for_status()
            
            # 保存音频
            temp_file = "/tmp/tts_output.wav"
            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 播放
            wf = wave.open(temp_file, 'rb')
            play_stream = self.audio.open(
                format=self.audio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            self.log("🔊 播放中...", "  ")
            data = wf.readframes(1024)
            while data and self.is_running:
                play_stream.write(data)
                data = wf.readframes(1024)
            
            play_stream.stop_stream()
            play_stream.close()
            wf.close()
            
            self.log("✓ 播放完成", "  ")
            
        except Exception as e:
            self.log(f"❌ TTS/播放错误: {e}", "  ")
        finally:
            # 恢复麦克风（延迟500ms，确保回声消散）
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
        english_text = self.process_audio(audio_data)
        if english_text is None:
            self.stats['failed'] += 1
            return self.is_running
        
        # 3. 播放
        self.play_tts(english_text, stream)
        
        # 统计
        elapsed = time.time() - start_time
        self.stats['processed'] += 1
        self.stats['avg_time'] = (
            (self.stats['avg_time'] * (self.stats['processed'] - 1) + elapsed)
            / self.stats['processed']
        )
        
        self.log(f"✓ 本轮完成 (耗时: {elapsed:.1f}s)", "  ")
        print()  # 空行分隔
        
        return self.is_running
    
    def start(self):
        """启动系统"""
        self.is_running = True
        
        # 打印欢迎
        print(f"\n{'='*60}")
        print("🌍 实时语音翻译系统 v4.0")
        print(f"{'='*60}\n")
        
        # 初始化模型
        self.init_whisper()
        
        # 打开麦克风
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
        
        # 校准环境噪音
        self.vad.calibrate(stream)
        
        print(f"\n{'='*60}")
        print("📋 系统就绪")
        print(f"   流程: 监听 → 录音 → 识别 → 翻译 → 播放")
        print(f"   配置: 静音 {SPEECH_END_SILENCE_MS}ms 结束 | 时长 {SPEECH_MIN_DURATION}-{SPEECH_MAX_DURATION}s")
        print(f"   操作: 正常说话即可 | Ctrl+C 退出")
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
            
            # 打印统计
            print(f"\n{'='*60}")
            print("📊 运行统计")
            print(f"   成功: {self.stats['processed']} | 失败: {self.stats['failed']}")
            if self.stats['processed'] > 0:
                print(f"   平均耗时: {self.stats['avg_time']:.1f}s")
            print(f"{'='*60}")
            self.log("系统已退出 👋")


if __name__ == "__main__":
    translator = RealtimeTranslator()
    translator.start()