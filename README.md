# Real-time Speech-to-Text using Faster-Whisper and a Web Interface

Capture live microphone audio, stream transcriptions to webpage in real time, and record audio as a WAV file.  


---

##  Features

-  Live microphone audio capture
-  Faster-Whisper transcription
-  Real-time streaming via WebSocket
-  Live transcription display in the browser
-  Session recording & WAV download


---

---

## ⚙ Requirements

### System

- Python 3.9+  
- Microphone input  

### GPU (This a important to use faster-whisper on gpu)

To use Faster-Whisper efficiently, install **PyTorch with CUDA support**. For example, if you have **CUDA 12.1**, install:

```bash
pip install torch==2.5.1+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

##  How It Works

The system pipeline:

### Backend (Python)

- Captures audio from the microphone in small chunks  
- Buffers audio for transcription windows (~5 seconds)  
- Runs Faster-Whisper to transcribe buffered audio  
- Sends transcription segments to connected browser clients  
- Records session audio and saves it as WAV  

### Frontend (HTML/JS)

- Connects to backend via WebSocket  
- Displays live transcriptions and segment timestamps  
- Tracks session statistics  
- Allows downloading recorded audio  

---




