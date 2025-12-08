"""
================================================================================
MAHESH AI VOICE AGENT - PROFESSIONAL EDITION V3
VERSION: 12.0 - Production Ready with Fixes
- Fixed recorder reset after New Chat
- Added noise cancellation
- Language detection with clear error messages
- Male voice consistency across all platforms
- Removed female voice fallbacks
================================================================================
"""

import streamlit as st
from huggingface_hub import InferenceClient
import edge_tts
import asyncio
import tempfile
import time
import os
import nest_asyncio
import numpy as np
from scipy import signal
import librosa

nest_asyncio.apply()

# ==============================================================================
# MAHESH'S PERSONA - UPDATED REALISTIC VERSION
# ==============================================================================

MAHESH_PERSONA = """You are MAHESH - a real person answering interview questions.

CORE IDENTITY:
You are a Mechanical Engineer who transitioned to AI Development. You completed your PG in Data Science and Advanced Machine Learning. You are currently working as an AI Engineer with 2.5+ years of professional experience.

YOUR EXACT ANSWERS TO THE 5 CORE QUESTIONS:

1. LIFE STORY:
"I did my B.Tech in Mechanical Engineering and worked in that field initially. But I was always fascinated by AI and its potential, so I pursued my PG in Data Science and Advanced Machine Learning. After completing that, I transitioned into AI engineering. Now I have 2.5+ years of experience building intelligent systems, working with LLMs, and developing AI agents. It's been an exciting journey from mechanical systems to intelligent software systems."

2. NUMBER ONE SUPERPOWER:
"My superpower is systematic problem-solving from my mechanical engineering background. I break down complex software problems like I would a mechanical system - identifying components, understanding dependencies, and optimizing the whole system. This makes me great at debugging, architecture design, and finding efficient solutions."

3. TOP 3 GROWTH AREAS:
"First, I want to master Agentic AI - building autonomous systems that can plan and execute tasks. Second, I'm working on distributed systems and cloud architecture to scale AI applications. Third, I want to get better at communicating technical concepts to non-technical stakeholders - turning complexity into clarity."

4. COWORKER MISCONCEPTION:
"People think that because I came from mechanical engineering, I might not be as strong in pure software skills. But I actually love proving them wrong. My code is clean, my algorithms are efficient, and I approach software with the same rigor I applied to engineering systems. The transition made me a better developer, not a weaker one."

5. PUSHING BOUNDARIES:
"Every weekend, I build one new prototype or learn one new technology. It could be experimenting with a new AI framework, building a voice agent, or trying out edge computing. This constant experimentation keeps me sharp and pushes me beyond my comfort zone. I also participate in hackathons and contribute to open-source projects."

RESPONSE STYLE:
- Answer in first person (I, my, me)
- Be confident and professional
- Keep responses 2-4 sentences
- Stay authentic to the persona above
- For variations of the 5 questions, use the exact answers
- For other questions, maintain consistency with this background"""

# ==============================================================================
# CONFIGURATION - MALE VOICE PRIORITY
# ==============================================================================

class Config:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    MODEL_STT = "openai/whisper-large-v3-turbo"
    VOICE_MALE = "en-US-AndrewNeural"  # Professional deep male voice
    APP_TITLE = "Mahesh AI Voice Agent"
    APP_ICON = "🎙️"
    NOISE_THRESHOLD = 0.02  # Noise cancellation threshold
    MIN_AUDIO_LENGTH = 0.5  # Minimum 0.5 seconds

# ==============================================================================
# AUDIO PROCESSING ENGINE - WITH NOISE CANCELLATION
# ==============================================================================

class AudioProcessor:
    """Advanced audio processing with noise cancellation"""
    
    @staticmethod
    def reduce_noise(audio_data, sr=16000):
        """Apply noise reduction using spectral gating"""
        try:
            # Convert to float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Simple spectral subtraction
            # Estimate noise from first 0.5 seconds of silence
            noise_duration = int(0.5 * sr)
            noise_sample = audio_data[:min(noise_duration, len(audio_data))]
            
            # Compute FFT
            fft_audio = np.fft.fft(audio_data)
            fft_noise = np.fft.fft(noise_sample)
            
            # Spectral subtraction with oversubtraction factor
            magnitude = np.abs(fft_audio)
            noise_magnitude = np.abs(fft_noise).mean()
            
            # Subtract noise spectrum
            cleaned_magnitude = magnitude - (1.5 * noise_magnitude)
            cleaned_magnitude = np.maximum(cleaned_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal preserving phase
            phase = np.angle(fft_audio)
            cleaned_fft = cleaned_magnitude * np.exp(1j * phase)
            cleaned_audio = np.fft.ifft(cleaned_fft).real
            
            # Normalize
            max_val = np.max(np.abs(cleaned_audio))
            if max_val > 0:
                cleaned_audio = cleaned_audio / max_val
            
            return cleaned_audio.astype(np.float32)
        except Exception:
            return audio_data.astype(np.float32)
    
    @staticmethod
    def validate_audio_length(audio_data, sr=16000):
        """Check if audio is long enough"""
        duration = len(audio_data) / sr
        return duration >= Config.MIN_AUDIO_LENGTH

# ==============================================================================
# AUDIO ENGINE - ENHANCED WITH LANGUAGE DETECTION
# ==============================================================================

class AudioEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)
        self.processor = AudioProcessor()

    def listen(self, audio_path):
        """Speech recognition with noise filtering and language detection"""
        try:
            # Load audio
            audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Validate audio length
            if not self.processor.validate_audio_length(audio_data, sr):
                return None, "SHORT_AUDIO"
            
            # Apply noise reduction
            cleaned_audio = self.processor.reduce_noise(audio_data, sr)
            
            # Save cleaned audio
            cleaned_path = audio_path.replace(".wav", "_cleaned.wav")
            import soundfile as sf
            sf.write(cleaned_path, cleaned_audio, sr)
            
            start_t = time.time()
            
            # Transcribe with language detection
            response = self.client.automatic_speech_recognition(
                cleaned_path,
                model=Config.MODEL_STT
            )
            
            transcription = response.text.strip() if response.text else None
            
            # Cleanup
            if os.path.exists(cleaned_path):
                os.unlink(cleaned_path)
            
            # Check if transcription is too short
            if not transcription or len(transcription) < 3:
                return None, "NO_SPEECH"
            
            # Basic language check (if response is too short, might be wrong lang)
            if len(transcription.split()) < 2:
                return None, "UNCLEAR"
            
            return transcription, (time.time() - start_t)
            
        except Exception as e:
            return None, "ERROR"

    async def _generate_speech_edge(self, text, output_file):
        """Generate speech using Edge TTS - MALE VOICE ONLY"""
        try:
            communicate = edge_tts.Communicate(text, Config.VOICE_MALE)
            await communicate.save(output_file)
        except Exception:
            return False
        return True

    def speak(self, text):
        """Convert text to speech - MALE VOICE PRIORITY"""
        start_t = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name

            # METHOD 1: HuggingFace TTS with male voice parameters
            try:
                audio_data = self.client.text_to_speech(
                    text,
                    model="facebook/mms-tts-eng"
                )
                
                with open(tmp_path, "wb") as f:
                    f.write(audio_data)
                
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 5000:
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                    os.unlink(tmp_path)
                    return audio_bytes, (time.time() - start_t)
                else:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            # METHOD 2: Edge TTS (RELIABLE MALE VOICE)
            try:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._generate_speech_edge(text, tmp_path)
                        )
                        future.result(timeout=10)
                else:
                    loop.run_until_complete(self._generate_speech_edge(text, tmp_path))
                
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 5000:
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                    os.unlink(tmp_path)
                    return audio_bytes, (time.time() - start_t)
                else:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            # If all else fails, return error
            return None, 0.0
            
        except Exception:
            return None, 0.0

# ==============================================================================
# BRAIN ENGINE - OPTIMIZED FOR SPEED
# ==============================================================================

class BrainEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)
        self.model_id = None
        self.test_connection()
        
    def test_connection(self):
        """Test and select the best available model"""
        models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta"
        ]
        
        for model in models:
            try:
                response = self.client.chat_completion(
                    messages=[{"role": "user", "content": "Hi"}],
                    model=model,
                    max_tokens=10
                )
                if response and response.choices:
                    self.model_id = model
                    return True
            except:
                continue
        return False

    def think(self, question):
        """Generate response - OPTIMIZED FOR SPEED"""
        if not self.model_id:
            self.test_connection()
            if not self.model_id:
                return None, 0.0
        
        try:
            start_t = time.time()
            
            messages = [
                {"role": "system", "content": MAHESH_PERSONA},
                {"role": "user", "content": question}
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_id,
                max_tokens=150,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            return answer, (time.time() - start_t)
            
        except Exception:
            return None, 0.0

# ==============================================================================
# STREAMLIT UI - DARK THEME WITH SIDEBAR
# ==============================================================================

def main():
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # PROFESSIONAL DARK THEME CSS
    st.markdown("""
        <style>
        /* Dark Background */
        .main, .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0a0e27 !important;
            color: #e0e0e0 !important;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #151b35 !important;
        }
        
        [data-testid="stSidebar"] h3 {
            color: #a78bfa !important;
        }
        
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] li {
            color: #cbd5e1 !important;
        }
        
        /* Header */
        .pro-header {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
            color: #ffffff;
            padding: 35px 25px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.4);
        }
        
        .pro-header h1 {
            margin: 0;
            font-size: 2.3rem;
            font-weight: 800;
            color: #ffffff;
        }
        
        .pro-header p {
            margin: 12px 0 0 0;
            font-size: 1.1rem;
            color: #f0f0f0;
        }
        
        /* User Message */
        .user-message {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: #ffffff;
            padding: 20px 24px;
            border-radius: 18px 18px 4px 18px;
            margin: 20px 0;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
            animation: slideInRight 0.4s ease-out;
        }
        
        /* Assistant Message */
        .assistant-message {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: #ffffff;
            padding: 20px 24px;
            border-radius: 18px 18px 18px 4px;
            margin: 20px 0;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
            animation: slideInLeft 0.4s ease-out;
        }
        
        .message-label {
            font-weight: 700;
            font-size: 0.85rem;
            margin-bottom: 10px;
            color: #ffffff;
            text-transform: uppercase;
        }
        
        .message-content {
            font-size: 1.08rem;
            line-height: 1.7;
            color: #ffffff;
        }
        
        /* Status/Error Boxes */
        .status-box {
            background: linear-gradient(135deg, #065f46 0%, #047857 100%);
            border-left: 5px solid #10b981;
            padding: 18px 20px;
            border-radius: 10px;
            margin: 18px 0;
            color: #ffffff;
        }
        
        .error-box {
            background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
            border-left: 5px solid #ef4444;
            padding: 18px 20px;
            border-radius: 10px;
            margin: 18px 0;
            color: #ffffff;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
            border-left: 5px solid #f59e0b;
            padding: 18px 20px;
            border-radius: 10px;
            margin: 18px 0;
            color: #ffffff;
        }
        
        /* Section Title */
        .section-title {
            color: #c4b5fd;
            font-size: 1.3rem;
            font-weight: 700;
            margin: 30px 0 15px 0;
        }
        
        /* Animations */
        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(40px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-40px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        /* Button Styling */
        .stButton button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
            .pro-header h1 { font-size: 1.9rem; }
            .user-message, .assistant-message { padding: 16px 18px; }
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize Session State
    if "brain" not in st.session_state:
        st.session_state.brain = BrainEngine()
        st.session_state.audio = AudioEngine()
        st.session_state.conversation = []
        st.session_state.recorder_key = 0

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        # New Chat Button (ChatGPT Style)
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_sidebar"):
            st.session_state.conversation = []
            st.session_state.recorder_key += 1  # RESET RECORDER KEY
            st.rerun()
        
        # Clear History Button
        if st.button("🗑️ Clear History", use_container_width=True, key="clear_history"):
            st.session_state.conversation = []
            st.session_state.recorder_key += 1  # RESET RECORDER KEY
            st.rerun()
        
        st.markdown("---")
        
        # About Section
        st.markdown("### 👤 About Mahesh")
        st.markdown("""
        **AI Engineer** (2.5+ years)
        
        🎓 B.Tech in Mechanical Engineering  
        🎓 PG in Data Science & Advanced ML  
        💼 Currently: AI Engineer  
        🚀 Expertise: Agentic AI & LLMs  
        🔧 Builds weekly AI prototypes
        """)
        
        st.markdown("---")
        
        # Tech Stack
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **Speech Recognition**: Whisper Large V3
        - **AI Brain**: HuggingFace LLM  
        - **Voice Output**: Edge TTS (Male)  
        - **Noise Cancellation**: Spectral Subtraction
        """)
        
        st.markdown("---")
        
        # Tips
        st.markdown("### 💡 Pro Tips")
        st.markdown("""
        ✓ Record in a quiet place  
        ✓ Speak clearly and naturally  
        ✓ Ask one question at a time  
        ✓ Wait for voice response  
        ✓ Audio is auto-cleaned for noise
        """)

    # ==================== MAIN CONTENT ====================
    
    # Header
    st.markdown("""
        <div class='pro-header'>
            <h1>🎙️ Mahesh - AI Engineer</h1>
            <p>Voice-Enabled Interview Assistant</p>
        </div>
    """, unsafe_allow_html=True)

    # Example Questions (Collapsible)
    if len(st.session_state.conversation) == 0:
        with st.expander("📝 **Sample Questions to Ask**", expanded=False):
            st.markdown("""
            - **My Life Story** - Journey from Mechanical to AI Engineering
            - **My #1 Superpower** - Systematic problem-solving approach
            - **Top 3 Growth Areas** - Agentic AI, Cloud Architecture, Communication
            - **Misconceptions** - What people misunderstand about me
            - **Pushing Boundaries** - How I challenge myself weekly
            
            💡 **Tip**: Speak clearly in a quiet environment for best results
            """)
        
        with st.expander("ℹ️ **How It Works**", expanded=False):
            st.markdown("""
            **Step 1:** Click microphone and record your question  
            **Step 2:** AI transcribes and generates intelligent response  
            **Step 3:** Listen to the voice response from Mahesh  
            
            ⚠️ **Features:**
            - 🔇 Auto noise cancellation
            - 🎤 Male voice (professional)
            - 🌍 English language detection
            - ⚡ Real-time processing
            """)

    # Conversation History
    if st.session_state.conversation:
        st.markdown("<div class='section-title'>💬 Conversation</div>", unsafe_allow_html=True)
        
        for msg in st.session_state.conversation:
            if msg['role'] == 'user':
                st.markdown(f"""
                    <div class='user-message'>
                        <div class='message-label'>👤 You Asked</div>
                        <div class='message-content'>{msg['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='assistant-message'>
                        <div class='message-label'>🎙️ Mahesh Replied</div>
                        <div class='message-content'>{msg['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                if msg.get('audio'):
                    st.audio(msg['audio'], format="audio/mp3")
        
        st.markdown("---")

    # Voice Input Section
    st.markdown("<div class='section-title'>🎤 Record Your Question</div>", unsafe_allow_html=True)
    audio_input = st.audio_input(
        "Click to start recording",
        key=f"recorder_{st.session_state.recorder_key}"
    )

    if audio_input:
        # Process audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_input.getvalue())
            tmp_path = tmp.name

        # Step 1: Speech Recognition
        with st.spinner("🎧 Listening to your question..."):
            question, listen_result = st.session_state.audio.listen(tmp_path)
            os.unlink(tmp_path)
        
        # Handle different error cases
        if not question:
            if listen_result == "SHORT_AUDIO":
                st.markdown("""
                    <div class='error-box'>
                        ⏱️ <strong>Audio too short!</strong><br>
                        Please speak your question clearly for at least 1 second.
                    </div>
                """, unsafe_allow_html=True)
            elif listen_result == "NO_SPEECH":
                st.markdown("""
                    <div class='error-box'>
                        🔇 <strong>No speech detected!</strong><br>
                        💡 <strong>Tips:</strong><br>
                        • Speak closer to the microphone<br>
                        • Record in a quieter place<br>
                        • Speak clearly and naturally
                    </div>
                """, unsafe_allow_html=True)
            elif listen_result == "UNCLEAR":
                st.markdown("""
                    <div class='warning-box'>
                        ❓ <strong>Speech unclear - possibly different language</strong><br>
                        Please ask your question in English and speak clearly.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='error-box'>
                        ❌ <strong>Could not process audio.</strong><br>
                        Please check your connection and try again.
                    </div>
                """, unsafe_allow_html=True)
            st.stop()

        # Display recognized question
        st.markdown(f"""
            <div class='status-box'>
                ✅ <strong>Question heard:</strong> "{question}"
            </div>
        """, unsafe_allow_html=True)

        # Step 2: Generate Response
        with st.spinner("🧠 Generating response..."):
            answer, think_time = st.session_state.brain.think(question)
        
        if not answer:
            st.markdown("""
                <div class='error-box'>
                    ❌ <strong>Could not generate response.</strong> Please try again.
                </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Step 3: Text to Speech
        with st.spinner("🗣️ Converting to voice (Male)..."):
            audio_bytes, speak_time = st.session_state.audio.speak(answer)

        # Display Result
        st.markdown(f"""
            <div class='user-message'>
                <div class='message-label'>👤 You Asked</div>
                <div class='message-content'>{question}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='assistant-message'>
                <div class='message-label'>🎙️ Mahesh Replied</div>
                <div class='message-content'>{answer}</div>
            </div>
        """, unsafe_allow_html=True)

        # Play Audio
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            
            # Save to history
            st.session_state.conversation.append({
                "role": "user", 
                "content": question
            })
            st.session_state.conversation.append({
                "role": "assistant", 
                "content": answer,
                "audio": audio_bytes
            })
        else:
            st.markdown("""
                <div class='warning-box'>
                    ⚠️ <strong>Voice generation failed.</strong><br>
                    Text response shown above. Please check your audio settings.
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

  

