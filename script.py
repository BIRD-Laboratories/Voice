import pyaudio
import numpy as np
import wave
import threading
import time
import queue
from scipy import signal
from scipy.io import wavfile
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from llama_cpp import Llama
import tempfile
import os

class SpeechChatbot:
    def __init__(self, record_duration=5.0, sample_rate=16000, chunk_size=1024):
        # Audio parameters
        self.record_duration = record_duration
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize audio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Audio buffers and state
        self.audio_buffer = []
        self.is_recording = False
        self.recording_start_time = None
        self.audio_queue = queue.Queue()
        
        # Initialize Whisper (using the base English model)
        print("Loading Whisper model...")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
        self.whisper_model.config.forced_decoder_ids = None
        
        # Initialize Llama
        print("Loading Llama model...")
        self.llm = Llama.from_pretrained(
            repo_id="unsloth/Qwen3-1.7B-GGUF",
            filename="Qwen3-1.7B-Q4_0.gguf",
            n_ctx=4096,  # Context window size
            verbose=False
        )
        
        # Conversation context
        self.conversation_context = []
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Always add to buffer if recording
        if self.is_recording:
            self.audio_buffer.append(in_data)
            
            # Check if recording duration has been reached
            if (time.time() - self.recording_start_time) > self.record_duration:
                self.is_recording = False
                print("Recording duration reached, processing audio...")
                
                # Save the audio buffer to process
                if self.audio_buffer:
                    audio_data = b''.join(self.audio_buffer)
                    self.audio_queue.put(audio_data)
                    self.audio_buffer = []
        
        return (in_data, pyaudio.paContinue)
    
    def save_audio_to_temp(self, audio_data):
        """Save audio data to temporary WAV file"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            return temp_file.name
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio using Whisper"""
        try:
            # Load and preprocess audio
            audio_array, sampling_rate = wavfile.read(audio_file_path)
            
            # Resample if necessary
            if sampling_rate != 16000:
                audio_array = signal.resample(audio_array, 
                                            int(len(audio_array) * 16000 / sampling_rate))
            
            # Process audio
            input_features = self.processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features
            
            # Generate transcription
            predicted_ids = self.whisper_model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            return transcription.strip()
        except Exception as e:
            print(f"Error in transcription: {e}")
            return None
    
    def process_with_llama(self, text):
        """Process text with Llama and maintain context"""
        if not text:
            return None
        
        # Add user message to context
        self.conversation_context.append({"role": "user", "content": text})
        
        # Keep only recent context to manage token limits
        if len(self.conversation_context) > 10:  # Keep last 10 exchanges
            self.conversation_context = self.conversation_context[-10:]
        
        try:
            # Generate response
            response = self.llm.create_chat_completion(
                messages=self.conversation_context,
                max_tokens=256,
                temperature=0.7,
                stop=["</s>", "\n\n"]
            )
            
            assistant_message = response['choices'][0]['message']['content']
            
            # Add assistant response to context
            self.conversation_context.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        except Exception as e:
            print(f"Error in LLM processing: {e}")
            return None
    
    def process_audio_worker(self):
        """Worker thread to process audio in background"""
        while True:
            try:
                audio_data = self.audio_queue.get(timeout=1.0)
                if audio_data is None:  # Shutdown signal
                    break
                
                # Save audio to temporary file
                temp_audio_file = self.save_audio_to_temp(audio_data)
                
                try:
                    # Transcribe audio
                    print("Transcribing audio...")
                    transcription = self.transcribe_audio(temp_audio_file)
                    
                    if transcription and len(transcription) > 2:  # Minimum length check
                        print(f"User: {transcription}")
                        
                        # Process with Llama
                        print("Processing with Llama...")
                        response = self.process_with_llama(transcription)
                        
                        if response:
                            print(f"Assistant: {response}")
                        else:
                            print("No response generated")
                    else:
                        print("No valid transcription detected")
                        
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_audio_file)
                    except:
                        pass
                    
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
    
    def start(self):
        """Start the chatbot"""
        print("Starting speech chatbot...")
        print(f"Recording duration: {self.record_duration} seconds")
        print("Press Enter to start recording, or Ctrl+C to stop...")
        
        # Start audio processing thread
        processing_thread = threading.Thread(target=self.process_audio_worker, daemon=True)
        processing_thread.start()
        
        # Start audio stream
        self.stream = self.audio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        self.stream.start_stream()
        
        try:
            while True:
                # Wait for user to press Enter to start recording
                input("Press Enter to start recording...")
                
                if not self.is_recording:
                    self.is_recording = True
                    self.recording_start_time = time.time()
                    self.audio_buffer = []
                    print(f"Recording for {self.record_duration} seconds...")
                    
        except KeyboardInterrupt:
            print("\nStopping chatbot...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the chatbot"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.audio_queue.put(None)  # Signal worker to stop

# Usage example
if __name__ == "__main__":
    chatbot = SpeechChatbot(
        record_duration=5.0,   # Record for 5 seconds
        sample_rate=16000,     # Whisper expects 16kHz
        chunk_size=1024        # Audio chunk size
    )
    
    chatbot.start()
