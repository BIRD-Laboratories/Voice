#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Voice Assistant — Press 'L' to Record
- Whisper (ONNX) for speech-to-text (Raspberry Pi optimized)
- Qwen3 (GGUF) via llama.cpp for responses
- Tkinter GUI (no browser, no Gradio)
- Press 'L' on keyboard to start/stop recording
"""

import os
import sys
import time
import threading
import logging
import tempfile
import wave
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | AI Assistant | %(message)s')
logger = logging.getLogger()

# ======================
# DEPENDENCIES
# ======================
try:
    import onnxruntime as ort
    from transformers import WhisperProcessor
    ONNX_AVAILABLE = True
except ImportError as e:
    ONNX_AVAILABLE = False
    logger.error(f"ONNX/Transformers import failed: {e}")

try:
    from llama_cpp import Llama
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    logger.error(f"llama_cpp import failed: {e}")

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

try:
    from pynput import keyboard
    from pynput.keyboard import Key, Listener
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.warning("pynput not installed. 'L' key shortcut disabled. Use button instead.")

# Audio recording
try:
    import pyaudio
    import scipy.io.wavfile as wavfile
    from scipy.signal import resample_poly
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.error("pyaudio or scipy not installed. Audio recording disabled.")


# ======================
# ONNX WHISPER
# ======================
class ONNXWhisper:
    def __init__(self, model_name="onnx-community/whisper-base"):
        from huggingface_hub import snapshot_download
        logger.info("Loading Whisper processor...")
        self.processor = WhisperProcessor.from_pretrained("onnx-community/whisper-base")
        logger.info("Downloading ONNX model (first run only)...")
        model_dir = snapshot_download(repo_id=model_name)
        model_path = Path(model_dir) / "model.onnx"
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        logger.info("✓ ONNX Whisper ready.")

    def transcribe(self, audio_array: np.ndarray, orig_sr: int) -> str:
        try:
            # Normalize to float32
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            elif audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # Stereo to mono
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)

            # Resample to 16kHz
            target_sr = 16000
            if orig_sr != target_sr:
                audio_array = resample_poly(audio_array, target_sr, orig_sr)

            # Process
            inputs = self.processor(audio_array, sampling_rate=target_sr, return_tensors="np")
            outputs = self.session.run(None, {"input_features": inputs.input_features})
            transcription = self.processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
            return transcription.strip()
        except Exception as e:
            return f"Transcribe error: {e}"


# ======================
# AUDIO RECORDER
# ======================
class AudioRecorder:
    def __init__(self, chunk=1024, channels=1, rate=16000):
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self.frames = []
        self.stream = None
        self.p = None
        self.is_recording = False

    def start(self):
        if not PYAUDIO_AVAILABLE:
            return False
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            self.frames = []
            self.is_recording = True
            threading.Thread(target=self._record_loop, daemon=True).start()
            return True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False

    def _record_loop(self):
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                logger.error(f"Recording error: {e}")
                break

    def stop(self):
        if not self.is_recording:
            return None
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

        # Save to temporary WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wf = wave.open(tmp.name, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # paInt16 = 2 bytes
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return tmp.name


# ======================
# AI ASSISTANT CORE
# ======================
class AIAssistant:
    def __init__(self):
        self.whisper = None
        self.llm = None
        self.conversation = []
        self.recorder = AudioRecorder()
        self._load_models()

    def _load_models(self):
        if ONNX_AVAILABLE:
            try:
                self.whisper = ONNXWhisper()
            except Exception as e:
                logger.error(f"Whisper ONNX load failed: {e}")
        if LLM_AVAILABLE:
            try:
                self.llm = Llama.from_pretrained(
                    repo_id="unsloth/Qwen3-1.7B-GGUF",
                    filename="Qwen3-1.7B-Q4_0.gguf",
                    n_ctx=2048,
                    n_threads=min(4, os.cpu_count() or 4),
                    verbose=False
                )
                logger.info("✓ Qwen3 loaded.")
            except Exception as e:
                logger.error(f"Qwen3 load failed: {e}")

    def transcribe_audio(self, wav_path: str) -> str:
        if not self.whisper:
            return "Whisper not available."
        try:
            orig_sr, audio_data = wavfile.read(wav_path)
            return self.whisper.transcribe(audio_data, orig_sr)
        except Exception as e:
            return f"Transcribe failed: {e}"

    def generate_response(self, text: str) -> str:
        if not self.llm or not text.strip():
            return "No response."
        self.conversation.append({"role": "user", "content": text})
        if len(self.conversation) > 6:
            self.conversation = self.conversation[-6:]
        try:
            resp = self.llm.create_chat_completion(
                messages=self.conversation,
                max_tokens=128,
                temperature=0.7
            )
            reply = resp['choices'][0]['message']['content'].strip()
            self.conversation.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"LLM error: {e}"

    def process_audio(self, wav_path: str):
        transcription = self.transcribe_audio(wav_path)
        if "error" in transcription.lower():
            return transcription, ""
        response = self.generate_response(transcription)
        return transcription, response


# ======================
# TKINTER GUI
# ======================
class VoiceAssistantGUI:
    def __init__(self, assistant: AIAssistant):
        self.assistant = assistant
        self.root = tk.Tk()
        self.root.title("🎙️ AI Voice Assistant")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Make window stay on top (optional)
        # self.root.attributes('-topmost', True)

        # Transcription label
        tk.Label(
            self.root, text="Transcription", font=("Arial", 16), bg="#f0f0f0", fg="#555"
        ).pack(pady=(20, 5))

        self.transcription_text = tk.Text(
            self.root, height=3, font=("Arial", 20, "bold"),
            wrap=tk.WORD, bg="#ffffff", fg="#333", relief=tk.SOLID, borderwidth=2
        )
        self.transcription_text.pack(padx=20, pady=5, fill=tk.X)

        # Response label
        tk.Label(
            self.root, text="AI Response", font=("Arial", 16), bg="#f0f0f0", fg="#555"
        ).pack(pady=(20, 5))

        self.response_text = tk.Text(
            self.root, height=5, font=("Arial", 24, "bold"),
            wrap=tk.WORD, bg="#e8f4fc", fg="#2c3e50", relief=tk.SOLID, borderwidth=2
        )
        self.response_text.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Press 'L' to start recording")
        self.status_label = tk.Label(
            self.root, textvariable=self.status_var, font=("Arial", 12),
            bg="#f0f0f0", fg="#777"
        )
        self.status_label.pack(pady=10)

        # Record button (fallback if pynput not used)
        self.record_btn = tk.Button(
            self.root, text="🎤 Record (Press 'L')", font=("Arial", 14),
            command=self.toggle_recording, bg="#3498db", fg="white",
            activebackground="#2980b9", height=2
        )
        self.record_btn.pack(pady=10)

        # Bind 'L' key in Tkinter (only works when window is focused)
        self.root.bind('<KeyPress-l>', lambda e: self.toggle_recording())
        self.root.bind('<KeyPress-L>', lambda e: self.toggle_recording())

        self.is_recording = False
        self.recording_start_time = None

        # Start global 'L' listener if pynput available
        if PYNPUT_AVAILABLE:
            self.start_global_listener()

    def start_global_listener(self):
        def on_press(key):
            try:
                if key.char and key.char.lower() == 'l':
                    self.root.after(0, self.toggle_recording)
            except AttributeError:
                pass  # Special keys

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not PYAUDIO_AVAILABLE:
            messagebox.showerror("Error", "Audio recording not available.")
            return
        if self.assistant.recorder.start():
            self.is_recording = True
            self.recording_start_time = time.time()
            self.record_btn.config(text="⏹️ Stop Recording", bg="#e74c3c")
            self.status_var.set("Recording... Press 'L' to stop.")
            self.transcription_text.delete(1.0, tk.END)
            self.response_text.delete(1.0, tk.END)
            logger.info("Recording started.")
        else:
            messagebox.showerror("Error", "Failed to start microphone.")

    def stop_recording(self):
        wav_path = self.assistant.recorder.stop()
        self.is_recording = False
        self.record_btn.config(text="🎤 Record (Press 'L')", bg="#3498db")
        self.status_var.set("Processing...")

        if not wav_path:
            self.status_var.set("No audio recorded.")
            return

        # Process in background
        def process():
            transcription, response = self.assistant.process_audio(wav_path)
            os.unlink(wav_path)  # Clean up temp file

            self.root.after(0, lambda: self.update_ui(transcription, response))

        threading.Thread(target=process, daemon=True).start()

    def update_ui(self, transcription: str, response: str):
        self.transcription_text.delete(1.0, tk.END)
        self.transcription_text.insert(tk.END, transcription)

        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, response)

        self.status_var.set("Press 'L' to record again.")

    def run(self):
        self.root.mainloop()


# ======================
# MAIN
# ======================
def main():
    logger.info("🚀 Starting Tkinter AI Voice Assistant (Raspberry Pi Optimized)...")

    if not TK_AVAILABLE:
        logger.critical("Tkinter not available. Install python3-tk.")
        return

    if not PYAUDIO_AVAILABLE:
        logger.critical("pyaudio required for recording. Install: pip install pyaudio")
        return

    assistant = AIAssistant()
    app = VoiceAssistantGUI(assistant)
    logger.info("GUI started. Press 'L' to record!")
    app.run()


if __name__ == "__main__":
    main()
