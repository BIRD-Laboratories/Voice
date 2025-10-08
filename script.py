#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Voice Assistant — Press 'L' to Record
- Whisper (OpenAI) for speech-to-text
- Qwen3 (Alibaba) for responses
- Press 'L' on keyboard to start/stop recording
- Large transcription display
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s | AI Assistant | %(message)s')
logger = logging.getLogger()

# ======================
# OPTIONAL DEPENDENCIES
# ======================
try:
    AI_MODELS_AVAILABLE = True
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from llama_cpp import Llama
except ImportError:
    AI_MODELS_AVAILABLE = False

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


# ======================
# DESKTOP ENVIRONMENT DETECTION (optional)
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
# AI ASSISTANT CORE
# ======================
class AIAssistant:
    def __init__(self):
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

    def transcribe(self, audio_path: str) -> str:
        if not (self.processor and self.whisper_model):
            return "ERROR: AI models not ready."

        try:
            import scipy.io.wavfile as wavfile
            from scipy.signal import resample_poly

            # Read audio (could be 44100, 48000, etc.)
            orig_sr, audio_array = wavfile.read(audio_path)

            # Convert to float32 if needed
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
            elif audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            # Handle stereo → mono
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)  # or use audio_array[:, 0]

            target_sr = 16000

            # Resample if necessary
            if orig_sr != target_sr:
                # Use resample_poly for high-quality resampling
                audio_array = resample_poly(audio_array, target_sr, orig_sr)

            # Now process with Whisper
            inputs = self.processor(
                audio_array,
                sampling_rate=target_sr,
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

    def chat(self, audio_path):
        if audio_path is None:
            return "No audio received.", ""
        try:
            transcription = self.transcribe(audio_path)
            if "ERROR" in transcription:
                return transcription, ""
            response = self.generate_response(transcription)
            return transcription, response
        except Exception as e:
            return f"Processing failed: {e}", ""


# ======================
# GRADIO INTERFACE
# ======================
def create_interface(ai_system: AIAssistant):
    with gr.Blocks(title="AI Voice Assistant") as demo:
        gr.Markdown("# 🎙️ AI Voice Assistant")
        gr.Markdown("### Press **'L'** on your keyboard to start/stop recording")

        with gr.Row():
            audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Microphone (press 'L' to toggle)",
                elem_id="mic_input"
            )
            with gr.Column():
                trans = gr.Textbox(
                    label="Transcription",
                    interactive=False,
                    elem_classes="transcription-box",
                    max_lines=3
                )
                resp = gr.Textbox(
                    label="AI Response",
                    interactive=False,
                    max_lines=5
                )

        # JavaScript to trigger mic when 'L' is pressed
        gr.HTML("""
        <script>
        document.addEventListener('keydown', function(event) {
            if (event.key.toLowerCase() === 'l') {
                event.preventDefault();
                const micBtn = document.querySelector('#mic_input button[aria-label="Record"]');
                if (micBtn) {
                    micBtn.click();
                    console.log("🎙️ 'L' pressed — toggling microphone");
                }
            }
        });

        // Optional: Add visual hint
        const style = document.createElement('style');
        style.textContent = `
            .transcription-box textarea {
                font-size: 28px !important;
                font-weight: bold;
                text-align: center;
                padding: 20px;
                height: 120px !important;
            }
        `;
        document.head.appendChild(style);
        </script>
        """)

        # Trigger on stop recording
        audio.stop_recording(
            fn=ai_system.chat,
            inputs=audio,
            outputs=[trans, resp],
            show_progress="minimal"
        )

    return demo


# ======================
# MAIN
# ======================
def main():
    logger.info("🚀 Starting AI Voice Assistant...")

    ai = AIAssistant()

    if not GRADIO_AVAILABLE:
        logger.critical("Gradio required. Install: pip install gradio")
        return

    PORT = 7860
    demo = create_interface(ai)

    def delayed_gui():
        time.sleep(2)
        launch_gui_demo(PORT)

    threading.Thread(target=delayed_gui, daemon=True).start()

    logger.info("Open http://localhost:7860 — then press 'L' to record!")
    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        show_error=True,
        quiet=True
    )


if __name__ == "__main__":
    main()
