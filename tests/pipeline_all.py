"""
最小化的实时翻译 Pipeline 测试
流程: 音频文件 → ASR → 翻译(HTTP API) → TTS(HTTP API) → 输出音频

使用前确保:
1. 翻译服务运行在 http://127.0.0.1:8099
2. TTS 服务运行在 http://127.0.0.1:9099 (pocket-tts serve --port 8000)
"""

import requests
from faster_whisper import WhisperModel
from pathlib import Path

# ============== 配置 ==============
TRANSLATION_API_URL = "http://127.0.0.1:8099/v1/chat/completions"
TTS_API_URL = "http://127.0.0.1:9099/tts"
INPUT_AUDIO = "./tests/pipeline_output.wav"
OUTPUT_AUDIO = "./tests/pipeline_output.wav"


# ============== ASR: 音频 → 中文文本 ==============
def asr_transcribe(audio_path: str) -> str:
    """使用 Faster-Whisper 将音频转为文本"""
    print(f"[ASR] 加载模型...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    print(f"[ASR] 识别音频: {audio_path}")
    segments, info = model.transcribe(audio_path, beam_size=5, language="zh")
    
    # 合并所有片段
    text = "".join(segment.text for segment in segments)
    print(f"[ASR] 识别结果: {text}")
    print(f"[ASR] 检测语言: {info.language} (置信度: {info.language_probability:.2f})")
    # text = "这是不是命运对我的惩罚？爱你也没办法？恨你也没办法"
    return text


# ============== 翻译: 中文 → 英文 (HTTP API) ==============
def translate_text(text: str) -> str:
    """使用翻译 API 将中文翻译为英文"""
    print(f"[翻译] 调用 API: {TRANSLATION_API_URL}")
    
    payload = {
        "model": "HY-MT1.5-1.8B",
        "messages": [
            {
                "role": "user",
                "content": f"Translate the following segment into English, without additional explanation.\n\n{text}"
            }
        ],
        "stream": False  # 非流式，简化处理
    }
    headers = {
        "Authorization": "Bearer sk-1234",
        "Content-Type": "application/json"
    }
    
    response = requests.post(TRANSLATION_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    
    result = response.json()
    # OpenAI 兼容格式
    translated = result["choices"][0]["message"]["content"]
    print(f"[翻译] 翻译结果: {translated}")
    return translated


# ============== TTS: 英文文本 → 音频 (HTTP API) ==============
def tts_generate(text: str, output_path: str) -> str:
    """使用 TTS API 将文本转为语音"""
    print(f"[TTS] 调用 API: {TTS_API_URL}")
    print(f"[TTS] 输入文本: {text}")
    
    # Pocket-TTS API 使用 multipart/form-data
    response = requests.post(
        TTS_API_URL,
        data={"text": text},
        stream=True  # 流式接收
    )
    response.raise_for_status()
    
    # 保存音频文件
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"[TTS] 音频已保存: {output_path}")
    return output_path


# ============== 主流程 ==============
def run_pipeline(input_audio: str, output_audio: str):
    """运行完整的翻译 pipeline"""
    print("=" * 50)
    print("实时翻译 Pipeline 测试")
    print("=" * 50)
    
    # 1. ASR: 音频 → 中文文本
    print("\n[步骤 1/3] ASR 语音识别")
    chinese_text = asr_transcribe(input_audio)
    
    if not chinese_text.strip():
        print("[错误] ASR 未识别到任何文本")
        return
    
    # 2. 翻译: 中文 → 英文
    print("\n[步骤 2/3] 翻译")
    english_text = translate_text(chinese_text)
    
    if not english_text.strip():
        print("[错误] 翻译结果为空")
        return
    
    # 3. TTS: 英文文本 → 音频
    print("\n[步骤 3/3] TTS 语音合成")
    tts_generate(english_text, output_audio)
    
    print("\n" + "=" * 50)
    print("Pipeline 完成!")
    print(f"输入音频: {input_audio}")
    print(f"识别文本: {chinese_text}")
    print(f"翻译结果: {english_text}")
    print(f"输出音频: {output_audio}")
    print("=" * 50)


if __name__ == "__main__":
    # 检查输入文件是否存在
    if not Path(INPUT_AUDIO).exists():
        print(f"[错误] 输入音频文件不存在: {INPUT_AUDIO}")
        print("请先准备一个中文音频文件")
        exit(1)
    
    run_pipeline(INPUT_AUDIO, OUTPUT_AUDIO)
