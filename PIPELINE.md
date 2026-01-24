# Realtrans Pipeline Documentation

## Overview

This project provides two real-time speech processing pipelines:

1. **Real-time Speech Translation** (`pipe_line_realtrans.py`) - Translates Chinese speech to English speech
2. **Real-time English Chat Partner** (`pipe_line_realchat.py`) - Conversational English practice assistant

---

## Pipeline Architecture

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   录音/     │  →   │   ASR/      │  →   │   LLM/      │  →   │   TTS/      │
│  Recording  │      │ Whisper     │      │  Qwen/      │      │  Kyutai     │
│   (VAD)     │      │  中文→文本   │      │  处理逻辑   │      │  文本→语音   │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
```

---

## Pipeline Comparison

| Feature | Realtrans (翻译) | Realchat (对话) |
|---------|-----------------|----------------|
| **Purpose** | Chinese → English translation | English conversation practice |
| **LLM Model** | HY-MT1.5-1.8B | Qwen3-0.6B |
| **LLM API** | http://127.0.0.1:8099/v1/chat/completions | http://127.0.0.1:6001/v1/chat/completions |
| **System Prompt** | Translation-focused | English partner helper |
| **History** | Stateless | Stateless |
| **Output** | Translated text | Conversational response |

---

## Component Details

### 1. Audio Recording (VAD)

**File**: Both pipelines

```python
# Uses WebRTC VAD with adaptive noise calibration
- Self-calibrating noise floor detection
- Dynamic volume threshold adjustment
- Speech start/end detection
```

**Key Parameters**:
- Sample Rate: 16kHz
- VAD Mode: 2 (aggressive)
- Speech Start: 10 consecutive frames (300ms)
- Speech End: 600ms silence
- Min Duration: 0.8s | Max Duration: 20s

---

### 2. ASR - Automatic Speech Recognition

**Test**: [tests/test_aasr.py](tests/test_aasr.py)

**Model**: faster-whisper (base)

**Configuration**:
```python
WhisperModel(
    model_size="base",
    device="cpu",
    compute_type="int8"
)
```

**Usage**:
```python
segments, info = model.transcribe(
    audio_file,
    beam_size=5,
    language="zh",
    vad_filter=True
)
```

---

### 3. LLM - Language Model

#### Translation Pipeline (Realtrans)

**Test**: [tests/test_llm.py](tests/test_llm.py)

**API Endpoint**: `http://127.0.0.1:8099/v1/chat/completions`

**Request Format**:
```python
{
    "model": "HY-MT1.5-1.8B",
    "messages": [{
        "role": "user",
        "content": f"Translate the following segment into English, without additional explanation.\n\n{text}"
    }],
    "stream": False
}
```

#### Chat Pipeline (Realchat)

**API Endpoint**: `http://127.0.0.1:6001/v1/chat/completions`

**System Prompt**:
```
You are a friendly English conversation partner helper.
The user is learning English and will speak in Chinese.
Respond naturally in English to help them practice.
Keep responses concise (1-2 sentences) and conversational.
Be encouraging and helpful.
```

**Request Format**:
```python
{
    "model": "Qwen3-0.6B/",
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ],
    "max_tokens": 256,
    "stream": False
}
```

---

### 4. TTS - Text to Speech

**Test**: [tests/test_tts.py](tests/test_tts.py)

**API Specification**: [models/tts_api.json](models/tts_api.json)

**API Endpoint**: `http://127.0.0.1:9099/tts`

**Request Format**:
```python
requests.post(
    "http://127.0.0.1:9099/tts",
    data={"text": text_to_speak}
)
```

---

## Running the Pipelines

### Prerequisites

```bash
# Install dependencies
pip install faster-whisper requests pyaudio webrtcvad numpy openai
```

### Start Services

```bash
# Start LLM service (for chat)
# Endpoint: http://localhost:6001/v1

# Start Translation service (for translation)
# Endpoint: http://localhost:8099/v1

# Start TTS service
# Endpoint: http://localhost:9099
```

### Run Pipelines

```bash
# Real-time Speech Translation (Chinese → English)
python pipe_line_realtrans.py

# Real-time English Chat Partner
python pipe_line_realchat.py
```

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          PIPELINE FLOW                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐          │
│  │  USER   │───▶│   VAD   │───▶│ ASR     │───▶│  LLM    │          │
│  │ SPEAKS  │    │ CALIB   │    │Whisper  │    │ Qwen    │          │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘          │
│       │                                            │               │
│       │           ┌─────────┐                      │               │
│       └──────────▶│ TTS     │◀─────────────────────┘               │
│                   │ Kyutai  │                                      │
│                   └─────────┘                                      │
│                       │                                            │
│                       ▼                                            │
│                   ┌─────────┐                                      │
│                   │  AUDIO  │                                      │
│                   │  OUTPUT │                                      │
│                   └─────────┘                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
realtrans/
├── pipe_line_realtrans.py    # Translation pipeline (中文→英文)
├── pipe_line_realchat.py     # Chat partner pipeline
├── models/
│   └── tts_api.json          # TTS API specification
├── tests/
│   ├── test_aasr.py          # Whisper ASR test
│   ├── test_llm.py           # LLM API test
│   └── test_tts.py           # TTS API test
└── PIPELINE.md               # This documentation
```

---

## Configuration Summary

| Service | Port | Model | Purpose |
|---------|------|-------|---------|
| LLM (Chat) | 6001 | Qwen3-0.6B | English conversation |
| LLM (Trans) | 8099 | HY-MT1.5-1.8B | Translation |
| TTS | 9099 | Kyutai | Text-to-Speech |
| ASR | Local | Whisper base | Speech-to-Text |
