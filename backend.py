import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import asyncio
import websockets
import json
import threading
from datetime import datetime
import wave
import os
import base64 

#set device="cpu" if you want to use it one cpu
model = WhisperModel("medium", device="cuda", compute_type="float16")

# Settings
samplerate = 16000 
blocksize = 4000  
q = queue.Queue()

connected_clients = set()

recorded_audio = []
session_start_time = None

# callback: puts audio chunks into the queue
def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    q.put(indata.copy())
    
    #store audio for recording
    recorded_audio.append(indata.copy())

#save audio as WAV file
def save_audio_recording():
    if recorded_audio:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        
        # Concatenate all audio chunks
        full_audio = np.concatenate(recorded_audio, axis=0)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(samplerate)
            
            # Convert float32 to int16
            audio_int16 = (full_audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        print(f"Audio saved as: {filename}")
        return filename
    return None

#function to send transcription to connected clients
async def broadcast_transcription(text, start_time, end_time):
    if connected_clients:
        message = json.dumps({
            'type': 'transcription',
            'text': text,
            'start': start_time,
            'end': end_time,
            'timestamp': datetime.now().isoformat()
        })
        
        disconnected = []
        for client in connected_clients.copy():
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(client)

        for client in disconnected:
            connected_clients.discard(client)

async def handle_client(websocket):
    connected_clients.add(websocket)
    print(f"Client connected. Total clients: {len(connected_clients)}")
    
    try:
        await websocket.send(json.dumps({
            'type': 'status',
            'message': 'Connected to transcription service'
        }))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
                elif data.get('type') == 'download_files':
                    audio_file = save_audio_recording()

                    audio_b64 = None
                    if audio_file and os.path.exists(audio_file):
                        with open(audio_file, "rb") as f:
                            audio_bytes = f.read()
                        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    
                    await websocket.send(json.dumps({
                        'type': 'files_ready',
                        'audio_file': audio_file,
                        'audio_data': audio_b64,  
                        'message': 'Audio file is ready for download'
                    }))
            except json.JSONDecodeError:
                pass
                
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"Client error: {e}")
    finally:
        connected_clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(connected_clients)}")

def transcription_worker():
    global session_start_time
    print("Starting transcription worker...")
    session_start_time = datetime.now()
    
    with sd.InputStream(samplerate=samplerate, channels=1, blocksize=blocksize, callback=callback):
        print("Listening for audio...")
        
        try:
            audio_buffer = np.zeros(0, dtype=np.float32)

            while True:
                data = q.get()
                data = data.flatten().astype(np.float32)
                audio_buffer = np.concatenate((audio_buffer, data))

                if len(audio_buffer) >= samplerate * 5:
                    segments, _ = model.transcribe(audio_buffer, beam_size=5,language="en")
                    
                    for segment in segments:
                        # if you want to print on console
                        #print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                        
                        # Send to web clients
                        asyncio.run_coroutine_threadsafe(
                            broadcast_transcription(segment.text, segment.start, segment.end),
                            loop
                        )
                    audio_buffer = audio_buffer[-samplerate * 2 :]

        except KeyboardInterrupt:
            print("\nTranscription stopped.")
            save_audio_recording()

# Main function
async def main():
    global loop
    loop = asyncio.get_event_loop()
    
    transcription_thread = threading.Thread(target=transcription_worker, daemon=True)
    transcription_thread.start()
    server = await websockets.serve(handle_client, "localhost", 8765)
    
    print("Server ready! Open the web page.")
    print("Press Ctrl+C to stop.")
    
    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        save_audio_recording()

if __name__ == "__main__":
    asyncio.run(main())
