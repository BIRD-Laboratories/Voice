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
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
        
        print("Loading Llama model...")
        self.llm = Llama.from_pretrained(
            repo_id="unsloth/Qwen3-1.7B-GGUF",
            filename="Qwen3-1.7B-Q4_0.gguf",
            n_ctx=4096,
            verbose=False
        )
        
        # Text-to-Speech
        print("Initializing Text-to-Speech...")
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.8)
        
        # Conversation context
        self.conversation_context = []
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if self.is_recording:
            self.audio_buffer.append(in_data)
            
            if (time.time() - self.recording_start_time) > self.record_duration:
                self.is_recording = False
                print("Recording complete, processing...")
                
                if self.audio_buffer:
                    audio_data = b''.join(self.audio_buffer)
                    self.audio_queue.put(audio_data)
                    self.audio_buffer = []
        
        return (in_data, pyaudio.paContinue)
    
    def save_audio_to_temp(self, audio_data):
        """Save audio data to proper WAV file"""
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
            # Load audio file properly
            with open(audio_file_path, "rb") as f:
                inputs = self.processor(
                    f.read(),
                    sampling_rate=16000,
                    return_tensors="pt"
                )
            
            input_features = inputs.input_features
            
            predicted_ids = self.whisper_model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            return transcription.strip()
        except Exception as e:
            print(f"Error in transcription: {e}")
            return None
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        if text:
            print(f"Assistant: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
    
    def process_with_llama(self, text):
        """Process text with Llama"""
        if not text:
            return None
        
        self.conversation_context.append({"role": "user", "content": text})
        
        # Keep context manageable
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
        
        try:
            response = self.llm.create_chat_completion(
                messages=self.conversation_context,
                max_tokens=256,
                temperature=0.7,
            )
            
            assistant_message = response['choices'][0]['message']['content']
            self.conversation_context.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        except Exception as e:
            print(f"Error in LLM processing: {e}")
            return None
    
    def process_audio_worker(self):
        """Worker thread to process audio"""
        while True:
            try:
                audio_data = self.audio_queue.get(timeout=1.0)
                if audio_data is None:
                    break
                
                temp_audio_file = self.save_audio_to_temp(audio_data)
                
                try:
                    print("Transcribing...")
                    transcription = self.transcribe_audio(temp_audio_file)
                    
                    if transcription and len(transcription) > 2:
                        print(f"User: {transcription}")
                        
                        print("Generating response...")
                        response = self.process_with_llama(transcription)
                        
                        if response:
                            self.text_to_speech(response)
                        else:
                            print("No response generated")
                    else:
                        print("No speech detected")
                        
                finally:
                    try:
                        os.unlink(temp_audio_file)
                    except:
                        pass
                    
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
    
    def start(self):
        """Start the chatbot"""
        print("Starting Speech Chatbot...")
        print(f"Record duration: {self.record_duration} seconds")
        print("Press Enter to record, Ctrl+C to exit")
        
        # Start processing thread
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
                input("Press Enter to start recording...")
                
                if not self.is_recording:
                    self.is_recording = True
                    self.recording_start_time = time.time()
                    self.audio_buffer = []
                    print(f"Recording for {self.record_duration} seconds...")
                    print("Speak now...")
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the chatbot"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        self.audio_queue.put(None)

if __name__ == "__main__":
    chatbot = SpeechChatbot(
        record_duration=5.0,
        sample_rate=16000,
        chunk_size=1024
    )
    
    chatbot.start()
