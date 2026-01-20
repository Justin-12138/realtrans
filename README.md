# RealTrans - Real-time Speech Translation System

A real-time speech translation system that transcribes Chinese speech, translates it to English, and synthesizes voice output.

## Features

- **Adaptive VAD (Voice Activity Detection)**: Dynamically adjusts volume thresholds for different environments
- **Intelligent Silence Detection**: Based on energy decay curves for accurate speech segmentation
- **In-memory Audio Processing**: Minimizes file I/O for better performance
- **Automatic Retry**: Handles network errors gracefully
- **Playback Isolation**: Disables microphone during TTS playback to prevent feedback
- **Progress Display**: Real-time feedback during recording and processing

## Pipeline

```
Microphone -> ASR (Whisper) -> Translation API -> TTS -> Speaker
    |            |                 |                |
    v            v                 v                v
  Audio      Chinese Text      English Text      Audio
```

## Requirements

- Python >= 3.12
- Linux (PortAudio dependency for PyAudio)

## Installation

```bash
# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt
```

## Dependencies

| Package | Purpose |
|---------|---------|
| faster-whisper | Speech recognition (Whisper model) |
| pyaudio | Audio input/output |
| requests | HTTP client for APIs |
| webrtcvad | Voice activity detection |
| numpy | Audio signal processing |
| scipy | Audio file I/O |
| torch | ML model backend |
| transformers | Translation model |
| pocket-tts | Text-to-speech (local editable) |

## External Services

The system requires two external services running:

### 1. Translation Service (HY-MT1.5-1.8B)

```bash
# Default endpoint: http://127.0.0.1:8099/v1/chat/completions
# OpenAI-compatible API format
```

### 2. TTS Service (Pocket-TTS)

```bash
# Start pocket-tts server
cd 3parts/pocket-tts
pocket-tts serve --port 9099

# Or use the API endpoint: http://127.0.0.1:9099/tts
```

## Usage

### Main Application

```bash
python main.py
```

**Workflow:**
1. System calibrates ambient noise (keep quiet for ~1.5s)
2. Wait for the "等待说话..." prompt
3. Speak in Chinese
4. System transcribes, translates, and plays English TTS
5. Press `Ctrl+C` to exit

### Test Scripts

```bash
# Test ASR only
python tests/test_aasr.py

# Test translation model only
python tests/test_translate.py

# Test TTS only
python tests/test_pockettts.py

# Test full pipeline with audio file
python tests/pipeline_all.py
```

## Configuration

Edit `main.py` to customize:

```python
# Audio parameters
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30

# VAD settings
VAD_MODE = 2
SPEECH_START_FRAMES = 10       # 300ms to trigger recording
SPEECH_END_SILENCE_MS = 600    # 600ms silence to stop
SPEECH_MIN_DURATION = 0.8      # Min recording length (seconds)
SPEECH_MAX_DURATION = 20       # Max recording length (seconds)

# API endpoints
TRANSLATION_API_URL = "http://127.0.0.1:8099/v1/chat/completions"
TTS_API_URL = "http://127.0.0.1:9099/tts"

# Retry settings
MAX_RETRIES = 2
TIMEOUT_TRANSLATION = 8
TIMEOUT_TTS = 15
```

## Project Structure

```
realtrans/
├── main.py                  # Main application
├── pyproject.toml           # Project config
├── tests/
│   ├── test_aasr.py         # ASR test
│   ├── test_translate.py    # Translation test
│   ├── test_pockettts.py    # TTS test
│   ├── pipeline_all.py      # Full pipeline test
│   └── pipeline_realtime.py # Duplicate of main.py
├── models/
│   ├── curl_translate.sh    # Translation API test
│   └── tts_api.json         # TTS API spec
└── 3parts/
    └── pocket-tts/          # Local TTS package (Kyutai)
```

## License

MIT
