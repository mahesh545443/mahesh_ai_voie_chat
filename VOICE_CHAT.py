"""
================================================================================
MAHESH AI VOICE AGENT - PROFESSIONAL EDITION V4
VERSION: 13.0 - All Issues Fixed
✅ Speed optimized (faster processing)
✅ Male voice fixed (reliable across all platforms)
✅ Consistent color scheme for all messages
✅ Better error handling
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
# MAHESH'S PERSONA
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
# CONFIGURATION - OPTIMIZED FOR SPEED AND RELIABILITY
# ==============================================================================

class Config:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    MODEL_STT = "openai/whisper-large-v3-turbo"
    
    # MALE VOICE OPTIONS (in priority order)
    VOICES_MALE = [
        "en-US-AndrewNeural",      # Deep professional male
        "en-US-GuyNeural",          # Clear male voice
        "en-GB-RyanNeural",         # British male
        "en-IN-PrabhatNeural"       # Indian English male
    ]
    
    APP_TITLE = "Mahesh AI Voice Agent"
    APP_ICON = "🎙️"
    NOISE_THRESHOLD = 0.02
    MIN_AUDIO_LENGTH = 0.5

# ==============================================================================
# AUDIO PROCESSING ENGINE - OPTIMIZED
# ==============================================================================

class AudioProcessor:
    """Fast audio processing with noise cancellation"""
    
    @staticmethod
    def reduce_noise(audio_data, sr=16000):
        """Quick noise reduction"""
        try:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Simple but fast noise reduction
            noise_duration = int(0.3 * sr)  # Reduced from 0.5s
            noise_sample = audio_data[:min(noise_duration, len(audio_data))]
            
            fft_audio = np.fft.fft(audio_data)
            fft_noise = np.fft.fft(noise_sample)
            
            magnitude = np.abs(fft_audio)
            noise_magnitude = np.abs(fft_noise).mean()
            
            cleaned_magnitude = magnitude - (1.2 * noise_magnitude)  # Reduced factor
            cleaned_magnitude = np.maximum(cleaned_magnitude, 0.1 * magnitude)
            
            phase = np.angle(fft_audio)
            cleaned_fft = cleaned_magnitude * np.exp(1j * phase)
            cleaned_audio = np.fft.ifft(cleaned_fft).real
            
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
# AUDIO ENGINE - SPEED OPTIMIZED WITH RELIABLE MALE VOICE
# ==============================================================================

class AudioEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)
        self.processor = AudioProcessor()

    def listen(self, audio_path):
        """Fast speech recognition"""
        try:
            audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            if not self.processor.validate_audio_length(audio_data, sr):
                return None, "SHORT_AUDIO"
            
            cleaned_audio = self.processor.reduce_noise(audio_data, sr)
            
            cleaned_path = audio_path.replace(".wav", "_cleaned.wav")
            import soundfile as sf
            sf.write(cleaned_path, cleaned_audio, sr)
            
            start_t = time.time()
            
            response = self.client.automatic_speech_recognition(
                cleaned_path,
                model=Config.MODEL_STT
            )
            
            transcription = response.text.strip() if response.text else None
            
            if os.path.exists(cleaned_path):
                os.unlink(cleaned_path)
            
            if not transcription or len(transcription) < 3:
                return None, "NO_SPEECH"
            
            if len(transcription.split()) < 2:
                return None, "UNCLEAR"
            
            return transcription, (time.time() - start_t)
            
        except Exception as e:
            return None, "ERROR"

    async def _generate_speech_edge(self, text, output_file, voice):
        """Generate speech with specified voice"""
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_file)
            return True
        except Exception:
            return False

    def speak(self, text):
        """RELIABLE MALE VOICE - Multiple fallback options"""
        start_t = time.time()
        
        # Try Edge TTS with multiple male voices
        for voice in Config.VOICES_MALE:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp_path = tmp.name
                
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
                            self._generate_speech_edge(text, tmp_path, voice)
                        )
                        success = future.result(timeout=8)  # Faster timeout
                else:
                    success = loop.run_until_complete(
                        self._generate_speech_edge(text, tmp_path, voice)
                    )
                
                if success and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 5000:
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                    os.unlink(tmp_path)
                    return audio_bytes, (time.time() - start_t), voice
                
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
            except Exception:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                continue
        
        # If all Edge TTS voices fail, try HuggingFace
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name
            
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
                return audio_bytes, (time.time() - start_t), "HuggingFace"
            else:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        return None, 0.0, None

# ==============================================================================
# BRAIN ENGINE - SPEED OPTIMIZED
# ==============================================================================

class BrainEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)
        self.model_id = None
        self.test_connection()
        
    def test_connection(self):
        """Test and select fastest model"""
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
        """Fast response generation"""
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
# STREAMLIT UI - CONSISTENT COLOR SCHEME
# ==============================================================================

def main():
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # CONSISTENT COLOR THEME CSS
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
        
        /* CONSISTENT MESSAGE COLORS - Same as header gradient */
        .user-message, .assistant-message {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
            color: #ffffff;
            padding: 20px 24px;
            border-radius: 18px;
            margin: 20px 0;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
        }
        
        .user-message {
            border-radius: 18px 18px 4px 18px;
            animation: slideInRight 0.4s ease-out;
        }
        
        .assistant-message {
            border-radius: 18px 18px 18px 4px;
            animation: slideInLeft 0.4s ease-out;
        }
        
        .message-label {
            font-weight: 700;
            font-size: 0.85rem;
            margin-bottom: 10px;
            color: #ffffff;
            text-transform: uppercase;
            opacity: 0.95;
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
        
        .success-box {
            background: linear-gradient(135deg, #065f46 0%, #047857 100%);
            border-left: 5px solid #10b981;
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
        
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_sidebar"):
            st.session_state.conversation = []
            st.session_state.recorder_key += 1
            st.rerun()
        
        if st.button("🗑️ Clear History", use_container_width=True, key="clear_history"):
            st.session_state.conversation = []
            st.session_state.recorder_key += 1
            st.rerun()
        
        st.markdown("---")
        
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
        
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **Speech Recognition**: Whisper V3 Turbo
        - **AI Brain**: Llama 3.2 / Mistral  
        - **Voice Output**: Edge TTS (Male)  
        - **Noise Filter**: Spectral Subtraction
        """)
        
        st.markdown("---")
        
        st.markdown("### 💡 Pro Tips")
        st.markdown("""
        ✓ Speak clearly in quiet place  
        ✓ Ask one question at a time  
        ✓ Wait for complete response  
        ✓ Male voice auto-selected  
        ✓ Noise auto-cleaned
        """)

    # ==================== MAIN CONTENT ====================
    
    st.markdown("""
        <div class='pro-header'>
            <h1>🎙️ Mahesh - AI Engineer</h1>
            <p>Voice-Enabled Interview Assistant</p>
        </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.conversation) == 0:
        with st.expander("📝 **Sample Questions to Ask**", expanded=False):
            st.markdown("""
            - **My Life Story** - Journey from Mechanical to AI Engineering
            - **My #1 Superpower** - Systematic problem-solving approach
            - **Top 3 Growth Areas** - Agentic AI, Cloud Architecture, Communication
            - **Misconceptions** - What people misunderstand about me
            - **Pushing Boundaries** - How I challenge myself weekly
            
            💡 **Tip**: Speak clearly for best results
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_input.getvalue())
            tmp_path = tmp.name

        # Step 1: Speech Recognition
        with st.spinner("🎧 Listening..."):
            question, listen_time = st.session_state.audio.listen(tmp_path)
            os.unlink(tmp_path)
        
        if not question:
            if listen_time == "SHORT_AUDIO":
                st.markdown("""
                    <div class='error-box'>
                        ⏱️ <strong>Audio too short!</strong><br>
                        Please speak for at least 1 second.
                    </div>
                """, unsafe_allow_html=True)
            elif listen_time == "NO_SPEECH":
                st.markdown("""
                    <div class='error-box'>
                        🔇 <strong>No speech detected!</strong><br>
                        • Speak closer to microphone<br>
                        • Find quieter location<br>
                        • Speak clearly
                    </div>
                """, unsafe_allow_html=True)
            elif listen_time == "UNCLEAR":
                st.markdown("""
                    <div class='warning-box'>
                        ❓ <strong>Speech unclear</strong><br>
                        Please ask in English and speak clearly.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='error-box'>
                        ❌ <strong>Processing error.</strong> Please try again.
                    </div>
                """, unsafe_allow_html=True)
            st.stop()

        st.markdown(f"""
            <div class='success-box'>
                ✅ <strong>Question heard:</strong> "{question}"
            </div>
        """, unsafe_allow_html=True)

        # Step 2: Generate Response
        with st.spinner("🧠 Thinking..."):
            answer, think_time = st.session_state.brain.think(question)
        
        if not answer:
            st.markdown("""
                <div class='error-box'>
                    ❌ <strong>Response generation failed.</strong> Try again.
                </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Step 3: Text to Speech
        with st.spinner("🗣️ Converting to voice (Male)..."):
            audio_bytes, speak_time, voice_used = st.session_state.audio.speak(answer)

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
            
            # Show voice info
            if voice_used:
                st.markdown(f"""
                    <div class='success-box'>
                        🎤 <strong>Voice used:</strong> {voice_used} (Male)
                    </div>
                """, unsafe_allow_html=True)
            
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
                <div class='error-box'>
                    ⚠️ <strong>Voice generation failed!</strong><br>
                    <strong>Possible solutions:</strong><br>
                    • Check your internet connection<br>
                    • Refresh the page and try again<br>
                    • Contact support if issue persists<br><br>
                    Text response is shown above.
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()



