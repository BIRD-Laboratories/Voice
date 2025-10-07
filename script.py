import pyaudio
import threading
import time
import queue
import tempfile
import os
import pyttsx3
import wave
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from llama_cpp import Llama

class SpeechChatbot:
    def __init__(self, record_duration=5.0, sample_rate=16000, chunk_size=1024):
        # Audio parameters
        self.record_duration = record_duration
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # Audio state
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = []
        self.is_recording = False
        self.recording_start_time = None
        self.audio_queue = queue.Queue()
        
        # Models
        print("Loading Whisper model...")
        try:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
            print("✓ Whisper model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Whisper model: {e}")
            return
        
        print("Loading Llama model...")
        try:
            self.llm = Llama.from_pretrained(
                repo_id="unsloth/Qwen3-1.7B-GGUF",
                filename="*Q4_0.gguf",  # Use wildcard to find any Q4_0 file
                n_ctx=4096,
                verbose=False
            )
            print("✓ Llama model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load Llama model: {e}")
            print("Please check if the model file exists")
            return
        
        # Text-to-Speech
        print("Initializing Text-to-Speech...")
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
            print("✓ TTS engine initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize TTS: {e}")
            return
        
        # Conversation context
        self.conversation_context = []
        print("✓ System ready for operation")
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if self.is_recording:
            self.audio_buffer.append(in_data)
            
            # Check if recording duration has been reached
            current_time = time.time()
            if (current_time - self.recording_start_time) >= self.record_duration:
                self.is_recording = False
                print("✓ Recording complete - processing audio...")
                
                # Save the audio buffer to process
                if self.audio_buffer:
                    audio_data = b''.join(self.audio_buffer)
                    self.audio_queue.put(audio_data)
                    self.audio_buffer = []  # Clear buffer for next recording
        
        return (in_data, pyaudio.paContinue)
    
    def save_audio_to_temp(self, audio_data):
        """Save audio data to proper WAV file"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data)
                return temp_file.name
        except Exception as e:
            print(f"✗ Error saving audio file: {e}")
            return None
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio using Whisper"""
        try:
            print("🔍 Starting transcription...")
            
            # Use the audio file directly
            with open(audio_file_path, "rb") as audio_file:
                inputs = self.processor(
                    audio_file.read(),
                    sampling_rate=16000,
                    return_tensors="pt"
                )
            
            # Generate transcription
            predicted_ids = self.whisper_model.generate(inputs.input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            print(f"✓ Transcription successful")
            return transcription.strip()
            
        except Exception as e:
            print(f"✗ Transcription failed: {e}")
            return None
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        if text and len(text.strip()) > 0:
            print(f"🗣️ Speaking response...")
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                print("✓ Speech completed")
            except Exception as e:
                print(f"✗ TTS error: {e}")
    
    def process_with_llama(self, text):
        """Process text with Llama"""
        if not text or len(text.strip()) < 2:
            return "I didn't catch that. Could you please repeat?"
        
        # Add user message to context
        self.conversation_context.append({"role": "user", "content": text})
        
        # Keep context manageable
        if len(self.conversation_context) > 8:
            self.conversation_context = self.conversation_context[-8:]
        
        try:
            print("🤖 Generating AI response...")
            response = self.llm.create_chat_completion(
                messages=self.conversation_context,
                max_tokens=150,
                temperature=0.7,
            )
            
            assistant_message = response['choices'][0]['message']['content'].strip()
            self.conversation_context.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except Exception as e:
            print(f"✗ LLM error: {e}")
            return "I'm having trouble processing that right now."
    
    def process_audio_worker(self):
        """Worker thread to process audio"""
        while True:
            try:
                audio_data = self.audio_queue.get(timeout=1.0)
                if audio_data is None:  # Shutdown signal
                    break
                
                # Save audio to temporary file
                temp_audio_file = self.save_audio_to_temp(audio_data)
                if not temp_audio_file:
                    continue
                
                try:
                    # Transcribe audio
                    transcription = self.transcribe_audio(temp_audio_file)
                    
                    if transcription and len(transcription.strip()) > 2:
                        print(f"👤 User said: '{transcription}'")
                        
                        # Process with LLM
                        response = self.process_with_llama(transcription)
                        
                        if response:
                            print(f"🤖 Assistant: {response}")
                            # Convert to speech
                            self.text_to_speech(response)
                        else:
                            self.text_to_speech("I didn't understand that.")
                    else:
                        print("No audible speech detected")
                        self.text_to_speech("I didn't hear anything. Please try again.")
                        
                except Exception as e:
                    print(f"✗ Processing error: {e}")
                    self.text_to_speech("System error. Please try again.")
                    
                finally:
                    # Clean up temporary file
                    try:
                        if os.path.exists(temp_audio_file):
                            os.unlink(temp_audio_file)
                    except:
                        pass
                
                self.audio_queue.task_done()
                print("\n✅ Ready for next command")
                
            except queue.Empty:
                continue
    
    def start(self):
        """Start the chatbot"""
        print("\n" + "="*50)
        print("🚀 PLA Speech Chatbot - System Online")
        print("="*50)
        print(f"⏱️  Recording duration: {self.record_duration} seconds")
        print("🎤 Press Enter to start recording")
        print("⏹️  Press Ctrl+C to shutdown system")
        print("="*50)
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_audio_worker, daemon=True)
        processing_thread.start()
        
        # Start audio stream
        try:
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            print("✓ Audio stream started successfully")
            
        except Exception as e:
            print(f"✗ Failed to start audio stream: {e}")
            print("Please check microphone permissions and connections")
            return
        
        # Main loop
        try:
            while True:
                input("\n🎤 Press Enter to record command...")
                
                if not self.is_recording:
                    self.is_recording = True
                    self.recording_start_time = time.time()
                    self.audio_buffer = []  # Clear any previous data
                    print(f"🔴 Recording... Speak now! ({self.record_duration}s)")
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Shutdown command received...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the chatbot"""
        print("🛑 Shutting down system...")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.audio_queue.put(None)  # Signal worker to stop
        print("✅ System shutdown complete")

if __name__ == "__main__":
    # Initialize and start the system
    chatbot = SpeechChatbot(
        record_duration=5.0,    # 5 second recordings
        sample_rate=16000,      # Whisper-compatible sample rate
        chunk_size=1024         # Standard chunk size
    )
    
    chatbot.start()
