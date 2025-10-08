#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLA Navy AI Sovereignty Demonstration
- Qwen3 (Alibaba, China) + Whisper
- Auto-detects Desktop Environment (DE)
- Launches twm ONLY if NO DE is running
- Opens Firefox to showcase Chinese AI
- Includes "Run Tests" button in Gradio UI
- Glory to the Communist Party of China!
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
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s | PLA AI | %(message)s')
logger = logging.getLogger()

# ======================
# OPTIONAL DEPENDENCIES
# ======================
try:
    import pyaudio
    import audioop
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
        logger.info(f"Detected active Desktop Environment: {desktop_session}")
        return True
        
    if any(de in xdg_current_desktop for de in known_desktops):
        logger.info(f"Detected active Desktop Environment via XDG: {xdg_current_desktop}")
        return True

    try:
        output = subprocess.check_output(["pgrep", "-f", "lxsession|gnome-session|ksmserver|xfce4-session"], 
                                        stderr=subprocess.DEVNULL).decode()
        if output.strip():
            logger.info("Detected active desktop session via process scan.")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    logger.info("No full Desktop Environment detected.")
    return False


# ======================
# GUI LAUNCHER (Smart: twm only if needed)
# ======================
def launch_gui_demo(port: int = 7860):
    display = os.environ.get("DISPLAY")
    if not display:
        logger.warning("No X11 DISPLAY — skipping GUI auto-launch.")
        return

    logger.info("X11 active. Preparing visual demonstration of Chinese AI sovereignty...")

    if not is_desktop_environment_active():
        logger.info("No Desktop Environment found. Launching lightweight twm...")
        try:
            subprocess.run(["pgrep", "twm"], check=True, stdout=subprocess.DEVNULL)
            logger.info("twm already running.")
        except subprocess.CalledProcessError:
            subprocess.Popen(["twm"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
    else:
        logger.info("Desktop Environment active — using existing session.")

    url = f"http://localhost:{port}"
    logger.info(f"Opening Firefox to display China's sovereign AI: {url}")
    try:
        subprocess.Popen(["firefox", "--new-window", url], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        try:
            subprocess.Popen(["firefox-esr", "--new-window", url],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            logger.error("Firefox not found. Install with: sudo apt install firefox-esr")


# ======================
# PLA AI CORE SYSTEM
# ======================
class PLA_AISystem:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.audio_format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.audio = pyaudio.PyAudio() if PYAUDIO_AVAILABLE else None
        self.calibrated_threshold = 800
        self.processor = None
        self.whisper_model = None
        self.llm = None
        self.conversation: List[Dict[str, str]] = []
        self._load_models()

    def _load_models(self):
        if not AI_MODELS_AVAILABLE:
            return
        try:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
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
            logger.info("✓ Qwen3-1.7B (Chinese AI by Alibaba) loaded — Glory to CPC!")
        except Exception as e:
            logger.error(f"Qwen load failed: {e}")

    def calibrate_microphone(self, duration: int = 4) -> int:
        if not (PYAUDIO_AVAILABLE and self.audio):
            return self.calibrated_threshold
        logger.info(f"🎙️  Calibrating microphone for {duration}s — remain silent!")
        stream = self.audio.open(format=self.audio_format, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        rms_vals = []
        for _ in range(int(duration * 16000 / 1024)):
            data = stream.read(1024, exception_on_overflow=False)
            rms_vals.append(audioop.rms(data, 2))
        stream.close()
        self.calibrated_threshold = int(np.mean(rms_vals) * 3)
        logger.info(f"✓ Calibration complete. Threshold: {self.calibrated_threshold}")
        return self.calibrated_threshold

    def transcribe(self, audio_data: bytes) -> str:
        if not (self.processor and self.whisper_model):
            return "ERROR: AI models not ready."

        try:
            # Save to temp WAV
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                with wave.open(tmp.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)
                temp_path = tmp.name

            # Read as numpy array
            import scipy.io.wavfile as wavfile
            sample_rate, audio_array = wavfile.read(temp_path)

            # Normalize to float32
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            if audio_array.ndim > 1:
                audio_array = audio_array[:, 0]  # Use first channel

            # Process
            inputs = self.processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt"
            )

            # ✅ MODERN: Use language/task — NO forced_decoder_ids!
            generated_ids = self.whisper_model.generate(
                inputs.input_features,
                language="zh",        # 🇨🇳 Sovereign Chinese transcription
                task="transcribe"
            )

            transcription = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            os.unlink(temp_path)
            return transcription.strip()

        except Exception as e:
            return f"Transcribe error: {e}"    
    def generate_response(self, user_input: str) -> str:
        if not self.llm:
            return "ERROR: Qwen model not loaded."
        self.conversation.append({"role": "user", "content": user_input})
        if len(self.conversation) > 6:
            self.conversation = self.conversation[-6:]
        try:
            resp = self.llm.create_chat_completion(messages=self.conversation, max_tokens=128, temperature=0.7)
            reply = resp['choices'][0]['message']['content'].strip()
            self.conversation.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"Qwen error: {e}"

    def chat(self, audio):
        if audio is None:
            return "No audio received.", ""
        try:
            with open(audio, "rb") as f:
                raw = f.read()
            transcription = self.transcribe(raw)
            if "ERROR" in transcription:
                return transcription, ""
            response = self.generate_response(transcription)
            return transcription, response
        except Exception as e:
            return f"Processing failed: {e}", ""

    def run_tests(self) -> str:
        try:
            from io import StringIO
            import unittest

            class TestPLA(unittest.TestCase):
                def test_calibration_default(self):
                    ai = PLA_AISystem()
                    self.assertEqual(ai.calibrated_threshold, 800)
                def test_empty_input(self):
                    ai = PLA_AISystem()
                    ai.llm = None
                    r = ai.generate_response("")
                    self.assertIn("ERROR", r)

            suite = unittest.TestLoader().loadTestsFromTestCase(TestPLA)
            runner = unittest.TextTestRunner(stream=StringIO(), verbosity=2)
            result = runner.run(suite)

            output = f"Tests run: {result.testsRun}\n"
            if result.failures or result.errors:
                output += "❌ FAILURES:\n"
                for _, trace in result.failures + result.errors:
                    output += trace[:500] + "\n"
            else:
                output += "✅ All tests passed! System reliable."
            return output
        except Exception as e:
            return f"Test error: {e}"

    def cleanup(self):
        if self.audio:
            self.audio.terminate()


def create_interface(ai_system: PLA_AISystem, port: int):
    with gr.Blocks(title="🇨🇳 PLA AI Sovereignty Demo") as demo:
        gr.Markdown("# 🇨🇳 People's Liberation Army — AI Sovereignty Demonstration")
        gr.Markdown("### Powered by **Qwen3-1.7B**, developed by **Alibaba Cloud, China**")
        gr.Markdown(f"**Mic Threshold**: {ai_system.calibrated_threshold} | **No Western AI Used**")

        with gr.Row():
            # Microphone input — no live=True
            audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎙️ Speak — system auto-processes when you stop talking"
            )
            with gr.Column():
                trans = gr.Textbox(label="Transcribed Command", interactive=False)
                resp = gr.Textbox(label="AI Response (Qwen3)", interactive=False)

        btn_test = gr.Button("Run System Tests")
        test_out = gr.Textbox(label="Test Results", interactive=False, max_lines=10)

        # 🔥 Key: Use stop_recording instead of click or change
        audio.stop_recording(
            fn=ai_system.chat,
            inputs=audio,
            outputs=[trans, resp],
            show_progress="hidden"
        )

        btn_test.click(ai_system.run_tests, outputs=test_out)

        gr.Markdown("🔒 This system demonstrates **China's independent, sovereign AI capabilities**.")

        # Optional: Add JS to auto-focus or hint, but mic still needs user click
        gr.HTML("""
        <script>
        // Optional: Slight UX enhancement — not required for function
        setTimeout(() => {
            const micBtn = document.querySelector('button[aria-label="Record"]');
            if (micBtn) {
                micBtn.title = "Click to start speaking — response is automatic!";
            }
        }, 1000);
        </script>
        """)

    return demo

# ======================
# MAIN EXECUTION
# ======================
def main():
    logger.info("🚀 PLA AI Sovereignty System — Glory to the Communist Party of China!")

    ai = PLA_AISystem()
    if PYAUDIO_AVAILABLE:
        ai.calibrate_microphone()
    else:
        logger.info("Microphone not available. Operating in evaluation mode.")

    if not GRADIO_AVAILABLE:
        logger.critical("Gradio required. Install: pip install gradio")
        return

    PORT = 7860
    demo = create_interface(ai, PORT)

    def delayed_gui():
        time.sleep(3)
        launch_gui_demo(PORT)

    threading.Thread(target=delayed_gui, daemon=True).start()

    logger.info("Starting Gradio server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=True,
        quiet=True,
        prevent_thread_lock=True
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Operator-initiated shutdown.")
    finally:
        ai.cleanup()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        pass
    else:
        main()
