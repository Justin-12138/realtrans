// English Chat Partner - Continuous Listening with VAD

class ContinuousChatApp {
    constructor() {
        this.audioContext = null;
        this.analyser = null;
        this.stream = null;
        this.mediaRecorder = null;
        this.audioChunks = [];

        // VAD configuration
        this.SAMPLE_RATE = 16000;
        this.SPEECH_START_FRAMES = 10;
        this.SPEECH_END_SILENCE_FRAMES = 20;
        this.SPEECH_MIN_DURATION = 800;  // ms
        this.SPEECH_MAX_DURATION = 20000;  // ms

        // VAD thresholds
        this.volumeThreshold = 300;
        this.calibrated = false;

        // State
        this.isListening = false;
        this.isRecording = false;
        this.speechBuffer = [];
        this.silenceFrames = 0;
        this.recordStartTime = 0;
        this.vadInterval = null;

        // DOM elements
        this.chatContainer = document.getElementById('chatContainer');
        this.statusDot = document.querySelector('.status-dot');
        this.statusText = document.querySelector('.status-text');
        this.recordingIndicator = document.getElementById('recordingIndicator');
        this.toggleBtn = document.getElementById('toggleBtn');
        this.waveformCanvas = document.getElementById('waveformCanvas');
        this.canvasCtx = this.waveformCanvas.getContext('2d');

        this.init();
    }

    async init() {
        // Setup toggle button
        this.toggleBtn.addEventListener('click', () => this.toggleListening());

        // Initialize canvas
        this.waveformCanvas.width = 300;
        this.waveformCanvas.height = 60;

        // Draw initial state
        this.drawWaveform();

        this.setStatus('ready', '点击按钮开始监听');
    }

    async toggleListening() {
        if (this.isListening) {
            this.stopListening();
        } else {
            await this.startListening();
        }
    }

    async startListening() {
        try {
            // Request microphone permission
            this.stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.SAMPLE_RATE,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            // Setup audio context for VAD
            this.audioContext = new AudioContext({ sampleRate: this.SAMPLE_RATE });
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;

            const source = this.audioContext.createMediaStreamSource(this.stream);
            source.connect(this.analyser);

            // Setup MediaRecorder for recording
            this.mediaRecorder = new MediaRecorder(this.stream, {
                mimeType: 'audio/webm'
            });

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = async () => {
                if (this.audioChunks.length > 0) {
                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                    this.audioChunks = [];
                    await this.sendAudio(audioBlob);
                }
            };

            this.isListening = true;
            this.toggleBtn.classList.add('active');
            this.toggleBtn.innerHTML = '<span class="icon"></span><span class="text">停止监听</span>';

            // Calibrate and start VAD
            await this.calibrate();
            this.startVAD();

        } catch (err) {
            console.error('Error starting listening:', err);
            this.setStatus('error', '无法访问麦克风');
        }
    }

    stopListening() {
        this.isListening = false;
        this.isRecording = false;

        if (this.vadInterval) {
            clearInterval(this.vadInterval);
            this.vadInterval = null;
        }

        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }

        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }

        if (this.audioContext) {
            this.audioContext.close();
        }

        this.toggleBtn.classList.remove('active');
        this.toggleBtn.innerHTML = '<span class="icon"></span><span class="text">开始监听</span>';
        this.recordingIndicator.classList.remove('active');
        this.setStatus('ready', '已暂停');
        this.speechBuffer = [];
    }

    async calibrate() {
        this.setStatus('busy', '校准中... (请保持安静)');

        const samples = [];
        const calibrationDuration = 1500;
        const startTime = Date.now();

        return new Promise((resolve) => {
            const calibrationInterval = setInterval(() => {
                if (Date.now() - startTime > calibrationDuration) {
                    clearInterval(calibrationInterval);

                    if (samples.length > 0) {
                        samples.sort((a, b) => a - b);
                        const idx = Math.floor(samples.length * 0.75);
                        this.noiseFloor = samples[idx];
                        this.volumeThreshold = Math.max(150, Math.min(800, this.noiseFloor * 2));
                        this.calibrated = true;
                        console.log(`Calibration: noiseFloor=${this.noiseFloor.toFixed(0)}, threshold=${this.volumeThreshold.toFixed(0)}`);
                    }

                    this.setStatus('ready', '正在监听...');
                    resolve();
                    return;
                }

                const dataArray = new Float32Array(this.analyser.fftSize);
                this.analyser.getFloatTimeDomainData(dataArray);
                const rms = this.calculateRMS(dataArray);
                samples.push(rms);
            }, 30);
        });
    }

    calculateRMS(data) {
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
            sum += data[i] * data[i];
        }
        return Math.sqrt(sum / data.length) * 32768;  // Scale to int16 range
    }

    startVAD() {
        this.vadInterval = setInterval(() => {
            if (!this.isListening) return;

            const dataArray = new Float32Array(this.analyser.fftSize);
            this.analyser.getFloatTimeDomainData(dataArray);
            const rms = this.calculateRMS(dataArray);
            const isSpeech = rms > this.volumeThreshold;

            // Update speech buffer
            this.speechBuffer.push({ isSpeech, rms, time: Date.now() });
            if (this.speechBuffer.length > this.SPEECH_START_FRAMES) {
                this.speechBuffer.shift();
            }

            if (!this.isRecording) {
                // Check if speech started
                if (this.speechBuffer.length === this.SPEECH_START_FRAMES) {
                    const speechCount = this.speechBuffer.filter(x => x.isSpeech).length;
                    if (speechCount >= this.SPEECH_START_FRAMES * 0.85) {
                        this.startRecording();
                    }
                }
            } else {
                // Recording in progress
                const elapsed = Date.now() - this.recordStartTime;

                if (isSpeech) {
                    this.silenceFrames = 0;
                } else {
                    this.silenceFrames++;
                }

                // Check for end conditions
                const shouldStop = (
                    this.silenceFrames >= this.SPEECH_END_SILENCE_FRAMES ||
                    elapsed >= this.SPEECH_MAX_DURATION
                );

                if (shouldStop && elapsed >= this.SPEECH_MIN_DURATION) {
                    this.stopRecording();
                } else if (elapsed >= this.SPEECH_MAX_DURATION) {
                    this.stopRecording();
                }

                // Update indicator
                this.updateRecordingIndicator(elapsed, isSpeech, rms);
            }

        }, 30);

        // Start waveform animation
        this.drawWaveform();
    }

    startRecording() {
        this.isRecording = true;
        this.audioChunks = [];
        this.silenceFrames = 0;
        this.recordStartTime = Date.now();

        // Start MediaRecorder
        if (this.mediaRecorder.state === 'inactive') {
            this.mediaRecorder.start(100);  // Chunk every 100ms
        }

        this.recordingIndicator.classList.add('active');
        this.setStatus('processing', '正在录音...');
    }

    stopRecording() {
        this.isRecording = false;
        this.recordingIndicator.classList.remove('active');

        // Stop MediaRecorder
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }

        this.setStatus('busy', '处理中...');
    }

    async convertToWav(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const audioContext = new AudioContext({ sampleRate: this.SAMPLE_RATE });
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

        const numberOfChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const bitDepth = 16;
        const bytesPerSample = bitDepth / 8;
        const blockAlign = numberOfChannels * bytesPerSample;

        const dataLength = audioBuffer.length * blockAlign;
        const bufferLength = 44 + dataLength;

        const arrayBufferNew = new ArrayBuffer(bufferLength);
        const view = new DataView(arrayBufferNew);

        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, numberOfChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, dataLength, true);

        const channelData = [];
        for (let i = 0; i < numberOfChannels; i++) {
            channelData.push(audioBuffer.getChannelData(i));
        }

        let offset = 44;
        for (let i = 0; i < audioBuffer.length; i++) {
            for (let channel = 0; channel < numberOfChannels; channel++) {
                const sample = Math.max(-1, Math.min(1, channelData[channel][i]));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                offset += 2;
            }
        }

        audioContext.close();
        return new Blob([arrayBufferNew], { type: 'audio/wav' });
    }

    async sendAudio(audioBlob) {
        try {
            const wavBlob = await this.convertToWav(audioBlob);
            const formData = new FormData();
            formData.append('audio', wavBlob, 'audio.wav');

            const response = await fetch('/api/chat', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Server error');
            }

            const data = await response.json();

            this.addMessage('user', data.user_text);
            this.addMessage('assistant', data.assistant_text);

            if (data.audio_base64) {
                await this.playAudio(data.audio_base64);
            }

            // Resume listening after audio plays
            if (this.isListening) {
                this.setStatus('ready', '正在监听...');
            }

        } catch (err) {
            console.error('Error:', err);
            this.addMessage('assistant', '抱歉，出错了。请再试一次。');
            this.setStatus('error', '出错了');
        }
    }

    async playAudio(hexString) {
        return new Promise((resolve) => {
            const bytes = new Uint8Array(hexString.length / 2);
            for (let i = 0; i < bytes.length; i++) {
                bytes[i] = parseInt(hexString.substr(i * 2, 2), 16);
            }

            const blob = new Blob([bytes], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);

            audio.onended = () => {
                URL.revokeObjectURL(audioUrl);
                resolve();
            };

            audio.onerror = () => {
                URL.revokeObjectURL(audioUrl);
                resolve();
            };

            audio.play();
        });
    }

    updateRecordingIndicator(elapsed, isSpeech, rms) {
        const indicator = this.recordingIndicator.querySelector('span:last-child');
        const duration = (elapsed / 1000).toFixed(1);
        const status = isSpeech ? '🔊' : '🔇';
        indicator.textContent = `${status} 录音中... ${duration}s | RMS: ${rms.toFixed(0)}`;
    }

    drawWaveform() {
        requestAnimationFrame(() => this.drawWaveform());

        const width = this.waveformCanvas.width;
        const height = this.waveformCanvas.height;

        this.canvasCtx.fillStyle = '#1a1a2e';
        this.canvasCtx.fillRect(0, 0, width, height);

        if (!this.isListening || !this.analyser) {
            this.canvasCtx.fillStyle = '#4a5568';
            this.canvasCtx.font = '12px sans-serif';
            this.canvasCtx.textAlign = 'center';
            this.canvasCtx.fillText('点击按钮开始', width / 2, height / 2);
            return;
        }

        const dataArray = new Float32Array(this.analyser.frequencyBinCount);
        this.analyser.getFloatTimeDomainData(dataArray);

        this.canvasCtx.lineWidth = 2;
        this.canvasCtx.strokeStyle = this.isRecording ? '#e53e3e' : '#48bb78';
        this.canvasCtx.beginPath();

        const sliceWidth = width / dataArray.length;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const v = dataArray[i] * height / 2;
            const y = height / 2 + v;

            if (i === 0) {
                this.canvasCtx.moveTo(x, y);
            } else {
                this.canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        this.canvasCtx.lineTo(width, height / 2);
        this.canvasCtx.stroke();
    }

    addMessage(type, text) {
        const welcome = this.chatContainer.querySelector('.welcome-message');
        if (welcome) {
            welcome.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        const textSpan = document.createElement('span');
        textSpan.className = 'message-text';
        textSpan.textContent = text;

        const timeSpan = document.createElement('span');
        timeSpan.className = 'message-time';
        timeSpan.textContent = new Date().toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit'
        });

        bubble.appendChild(textSpan);
        bubble.appendChild(timeSpan);
        messageDiv.appendChild(bubble);
        this.chatContainer.appendChild(messageDiv);

        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }

    setStatus(state, text) {
        this.statusText.textContent = text;
        this.statusDot.className = 'status-dot';

        if (state === 'ready') {
            this.statusDot.style.background = 'var(--success-color)';
        } else if (state === 'busy') {
            this.statusDot.style.background = 'var(--danger-color)';
        } else if (state === 'processing') {
            this.statusDot.className = 'status-dot processing';
        } else {
            this.statusDot.style.background = 'var(--danger-color)';
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ContinuousChatApp();
});
