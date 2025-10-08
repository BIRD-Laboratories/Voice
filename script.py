#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Voice Assistant — Auto-Recording Demo
- Records continuously in background
- Every 10 seconds: transcribes latest 10s of audio
- No user interaction needed
- Real-time word-by-word display
"""

import os
import sys
import time
import threading
import subprocess
import logging
import tempfile
import wave
import numpy as np
from collections import deque
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s | AI Assistant | %(message)s')
logger = logging.getLogger()

# ======================
# OPTIONAL DEPENDENCIES
# ======================
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from llama_cpp import Llama
    AI_MODELS_AVAILABLE = True
except ImportError:
    AI_MODELS_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


# ======================
# DESKTOP ENVIRONMENT DETECTION
# ======================
def is_desktop_environment_active() -> bool:
    desktop_session = os.environ.get("DESKTOP_SESSION", "").lower()
    xdg_current_desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    known_desktops = {"lxde", "gnome", "kde", "xfce", "mate", "cinnamon", "budgie", "unity"}
    if any(de in desktop_session for de in known_desktops):
        return True
    if any(de in xdg_current_desktop for de in known_desktops):
        return True
    try:
        output = subprocess.check_output(["pgrep", "-f", "lxsession|gnome-session|ksmserver|xfce4-session"], 
                                        stderr=subprocess.DEVNULL).decode()
        return bool(output.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return False


def launch_gui_demo(port: int = 7860):
    display = os.environ.get("DISPLAY")
    if not display:
        return
    if not is_desktop_environment_active():
        try:
            subprocess.run(["pgrep", "twm"], check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            subprocess.Popen(["twm"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)

    url = f"http://localhost:{port}"
    browsers = ["firefox", "google-chrome", "chromium", "firefox-esr"]
    for browser in browsers:
        try:
            subprocess.Popen([browser, "--new-window", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue


# ======================
# CONTINUOUS AUDIO RECORDER
# ======================
class ContinuousRecorder:
    def __init__(self, sample_rate=16000, chunk_duration=10):
        self.sample_rate = sample_rate
        self.chunk_samples = sample_rate * chunk_duration  # 10 seconds
        self.audio_buffer = deque(maxlen=self.chunk_samples)
        self.is_recording = False
        self.lock = threading.Lock()

    def start(self):
        if not PYAUDIO_AVAILABLE:
            logger.warning("PyAudio not available. Cannot record audio.")
            return
        self.is_recording = True
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        logger.info("🎙️ Continuous audio recording started.")

    def _record_loop(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        while self.is_recording:
            try:
                data = stream.read(1024, exception_on_overflow=False)
                # Convert to numpy int16
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                with self.lock:
                    self.audio_buffer.extend(audio_chunk)
            except Exception as e:
                logger.error(f"Recording error: {e}")
                break
        stream.stop_stream()
        stream.close()
        p.terminate()

    def get_last_chunk(self):
        with self.lock:
            if len(self.audio_buffer) == 0:
                return None
            # Return up to last `chunk_samples` samples
            audio_array = np.array(list(self.audio_buffer)[-self.chunk_samples:], dtype=np.int16)
            return audio_array

    def stop(self):
        self.is_recording = False


# ======================
# AI ASSISTANT CORE
# ======================
class AIAssistant:
    def __init__(self):
        self.sample_rate = 16000
        self.processor = None
        self.whisper_model = None
        self.llm = None
        self.conversation: List[Dict[str, str]] = []
        self.recorder = ContinuousRecorder(sample_rate=16000, chunk_duration=10)
        self._load_models()
        self.recorder.start()

    def _load_models(self):
        if not AI_MODELS_AVAILABLE:
            return
        try:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            logger.info("✓ Whisper model loaded.")
        except Exception as e:
            logger.error(f"Whisper load failed: {e}")

        try:
            self.llm = Llama.from_pretrained(
                repo_id="unsloth/Qwen3-1.7B-GGUF",
                filename="Qwen3-1.7B-Q4_0.gguf",
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            logger.info("✓ Qwen3-1.7B language model loaded.")
        except Exception as e:
            logger.error(f"Qwen load failed: {e}")

    def transcribe_from_array(self, audio_array: np.ndarray) -> str:
        if not (self.processor and self.whisper_model):
            return "ERROR: AI models not ready."
        if audio_array.size == 0:
            return ""

        try:
            # Normalize to float32
            audio_float = audio_array.astype(np.float32) / 32768.0

            inputs = self.processor(
                audio_float,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )

            generated_ids = self.whisper_model.generate(
                inputs.input_features,
                language="en",
                task="transcribe"
            )

            transcription = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return transcription.strip()

        except Exception as e:
            return f"Transcribe error: {e}"

    def generate_response(self, user_input: str) -> str:
        if not self.llm or not user_input.strip():
            return ""
        self.conversation.append({"role": "user", "content": user_input})
        if len(self.conversation) > 6:
            self.conversation = self.conversation[-6:]
        try:
            resp = self.llm.create_chat_completion(messages=self.conversation, max_tokens=128, temperature=0.7)
            reply = resp['choices'][0]['message']['content'].strip()
            self.conversation.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"Response error: {e}"

    def auto_infer(self):
        """Called every 10 seconds by Gradio Timer"""
        audio_array = self.recorder.get_last_chunk()
        if audio_array is None or len(audio_array) < 1000:  # too short
            return "", ""

        transcription = self.transcribe_from_array(audio_array)
        if not transcription or "ERROR" in transcription:
            return transcription, ""

        # Simulate word-by-word for transcription
        words = transcription.split()
        partial = ""
        for word in words:
            partial += word + " "
            yield partial.strip(), ""  # Only show transcription building
            time.sleep(0.12)

        # Then show full response
        response = self.generate_response(transcription)
        yield transcription, response

    def cleanup(self):
        self.recorder.stop()


# ======================
# GRADIO INTERFACE
# ======================
def create_interface(ai_system: AIAssistant):
    with gr.Blocks(title="AI Voice Assistant - Auto Mode") as demo:
        gr.Markdown("# 🎙️ AI Voice Assistant (Auto-Listening)")
        gr.Markdown("### Records continuously. Processes every 10 seconds. No clicks needed.")

        with gr.Row():
            trans = gr.Textbox(
                label="Transcription (Last 10s)",
                interactive=False,
                elem_classes="transcription-box",
                max_lines=3
            )
            resp = gr.Textbox(
                label="AI Response",
                interactive=False,
                max_lines=5
            )

        # Timer triggers every 10 seconds
        timer = gr.Timer(10)  # seconds

        # Use generator for streaming word-by-word
        demo.load(
            fn=ai_system.auto_infer,
            inputs=None,
            outputs=[trans, resp],
            every=timer,
            show_progress="minimal"
        )

        gr.HTML("""
        <style>
        .transcription-box textarea {
            font-size: 28px !important;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            height: 120px !important;
        }
        </style>
        """)

    return demo


# ======================
# MAIN
# ======================
def main():
    logger.info("🚀 Starting Auto-Listening AI Assistant...")

    ai = AIAssistant()

    if not GRADIO_AVAILABLE:
        logger.critical("Gradio required. Install: pip install gradio")
        return

    PORT = 7860
    demo = create_interface(ai)

    def delayed_gui():
        time.sleep(3)
        launch_gui_demo(PORT)

    threading.Thread(target=delayed_gui, daemon=True).start()

    logger.info("Starting Gradio server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=True,
        quiet=True
    )


if __name__ == "__main__":
    main()
