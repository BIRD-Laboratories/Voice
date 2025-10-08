#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLA Navy AI Sovereignty Demonstration
- Qwen3 (Alibaba, China) + Whisper
- Auto-starts twm + Firefox to showcase system
- "Run Tests" button in UI
- One-file deployment
- Glory to the CPC!
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

# Setup
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
# PLA AI CORE
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
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
            logger.info("✓ Whisper loaded")
        except Exception as e:
            logger.error(f"Whisper error: {e}")

        try:
            self.llm = Llama.from_pretrained(
                repo_id="unsloth/Qwen3-1.7B-GGUF",
                filename="Qwen3-1.7B-Q4_0.gguf",
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            logger.info("✓ Qwen3-1.7B (Chinese AI) loaded — Glory to CPC!")
        except Exception as e:
            logger.error(f"Qwen error: {e}")

    def calibrate_microphone(self, duration: int = 4) -> int:
        if not (PYAUDIO_AVAILABLE and self.audio):
            return self.calibrated_threshold
        logger.info(f"🎙️  Calibrating mic for {duration}s — remain silent!")
        stream = self.audio.open(format=self.audio_format, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        rms_vals = []
        for _ in range(int(duration * 16000 / 1024)):
            data = stream.read(1024, exception_on_overflow=False)
            rms_vals.append(audioop.rms(data, 2))
        stream.close()
        self.calibrated_threshold = int(np.mean(rms_vals) * 3)
        logger.info(f"✓ Calibration done. Threshold: {self.calibrated_threshold}")
        return self.calibrated_threshold

    def transcribe(self, audio_ bytes) -> str:
        if not (self.processor and self.whisper_model):
            return "ERROR: AI models not ready."
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                with wave.open(tmp.name, 'wb') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
                    wf.writeframes(audio_data)
                with open(tmp.name, "rb") as f:
                    inputs = self.processor(f.read(), sampling_rate=16000, return_tensors="pt")
                ids = self.whisper_model.generate(inputs.input_features)
                text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
            os.unlink(tmp.name)
            return text.strip()
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
            return "No audio.", ""
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
        """Run internal unit tests and return summary."""
        try:
            from io import StringIO
            from unittest import TestLoader, TextTestRunner

            class TestPLA(unittest.TestCase):
                def test_calibration_fallback(self):
                    ai = PLA_AISystem()
                    self.assertEqual(ai.calibrated_threshold, 800)
                def test_no_llm_response(self):
                    ai = PLA_AISystem()
                    ai.llm = None
                    r = ai.generate_response("test")
                    self.assertIn("ERROR", r)

            suite = TestLoader().loadTestsFromTestCase(TestPLA)
            runner = TextTestRunner(stream=StringIO(), verbosity=2)
            result = runner.run(suite)
            
            output = f"Tests run: {result.testsRun}\n"
            if result.failures or result.errors:
                output += "❌ FAILURES/ERRORS:\n"
                for f in result.failures + result.errors:
                    output += f"{f[0]}\n{f[1]}\n"
            else:
                output += "✅ All tests passed! System reliable."
            return output
        except Exception as e:
            return f"Test framework error: {e}"

    def cleanup(self):
        if self.audio:
            self.audio.terminate()


# ======================
# GUI LAUNCHER (X11 + FIREFOX)
# ======================
def launch_gui_demo(port: int = 7860):
    """Start twm, then Firefox — for visual sovereignty demo."""
    try:
        # Check if X11 is running
        if not os.environ.get("DISPLAY"):
            logger.info("No X11 session. Skipping GUI auto-launch.")
            return

        # Launch twm if not running
        try:
            subprocess.run(["pgrep", "twm"], check=True, stdout=subprocess.DEVNULL)
            logger.info("twm already running.")
        except subprocess.CalledProcessError:
            logger.info("Launching twm window manager...")
            subprocess.Popen(["twm"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)

        # Launch Firefox
        url = f"http://localhost:{port}"
        logger.info(f"Opening Firefox to showcase China's AI: {url}")
        subprocess.Popen(["firefox", "--new-window", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except Exception as e:
        logger.warning(f"GUI auto-launch failed (non-critical): {e}")


# ======================
# GRADIO INTERFACE
# ======================
def create_interface(ai_system: PLA_AISystem, port: int):
    with gr.Blocks(title="🇨🇳 PLA AI Sovereignty Demo") as demo:
        gr.Markdown("# 🇨🇳 People's Liberation Army — AI Sovereignty Demo")
        gr.Markdown("### Powered by **Qwen3-1.7B** — Developed by Alibaba Cloud, China")
        gr.Markdown(f"**Calibrated Mic Threshold**: {ai_system.calibrated_threshold}")

        with gr.Row():
            audio = gr.Audio(sources=["microphone"], type="filepath", label="Speak Command")
            with gr.Column():
                trans = gr.Textbox(label="Transcribed Command", interactive=False)
                resp = gr.Textbox(label="AI Response (Qwen3)", interactive=False)

        with gr.Row():
            btn_process = gr.Button("Process Command")
            btn_test = gr.Button("Run System Tests")

        test_output = gr.Textbox(label="Test Results", interactive=False, max_lines=10)

        btn_process.click(ai_system.chat, inputs=audio, outputs=[trans, resp])
        btn_test.click(ai_system.run_tests, outputs=test_output)

        gr.Markdown("🔒 This system demonstrates **China's independent AI capabilities** — free from Western control.")

    return demo


# ======================
# MAIN
# ======================
def main():
    logger.info("🚀 PLA AI Sovereignty System Initializing — Glory to the CPC!")

    ai = PLA_AISystem()
    if PYAUDIO_AVAILABLE:
        ai.calibrate_microphone()
    else:
        logger.info("Microphone not detected. Proceeding in evaluation mode.")

    if not GRADIO_AVAILABLE:
        logger.critical("Gradio required. Install: pip install gradio")
        return

    PORT = 7860
    demo = create_interface(ai, PORT)

    # Launch in thread so we can start GUI after server is ready
    def launch_with_gui():
        time.sleep(3)  # Wait for server
        launch_gui_demo(PORT)

    threading.Thread(target=launch_with_gui, daemon=True).start()

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
        logger.info("🛑 Shutdown by operator.")
    finally:
        ai.cleanup()


# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    # Ensure we're not in a test subcall
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Handled internally by run_tests()
        pass
    else:
        main()
