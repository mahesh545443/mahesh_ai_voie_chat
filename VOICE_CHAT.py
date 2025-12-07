"""
================================================================================
MAHESH AI VOICE AGENT - PROFESSIONAL EDITION
VERSION: 10.0 - Production Ready
AUTHOR: Mahesh
================================================================================
"""

import streamlit as st
from huggingface_hub import InferenceClient
import edge_tts
from gtts import gTTS
import asyncio
import tempfile
import time
import os
import nest_asyncio

nest_asyncio.apply()

# ==============================================================================
# MAHESH'S PERSONA - STRUCTURED FOR ACCURACY
# ==============================================================================

MAHESH_PERSONA = """You are MAHESH - a real person answering interview questions.

CORE IDENTITY:
You are a Mechanical Engineer who transitioned to AI Development. You have a PG in Data Science and Machine Learning. You combine engineering rigor with cutting-edge AI expertise.

YOUR EXACT ANSWERS TO THE 5 CORE QUESTIONS:

1. LIFE STORY:
"I started my career as a Mechanical Engineer, working on manufacturing systems and automation. But I realized the future was in software, not just mechanics. So I spent my nights learning Python, building small projects, and slowly transitioning into AI development. Now I work on building intelligent agents and optimizing LLM systems - combining my engineering mindset with cutting-edge AI."

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
# CONFIGURATION
# ==============================================================================

class Config:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    MODEL_STT = "openai/whisper-large-v3-turbo"
    VOICE_MALE = "en-US-ChristopherNeural"
    APP_TITLE = "Mahesh AI Voice Agent"
    APP_ICON = "🎙️"

# ==============================================================================
# ENHANCED AUDIO ENGINE
# ==============================================================================

class AudioEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)

    def listen(self, audio_path):
        """Enhanced speech recognition with better accuracy"""
        try:
            start_t = time.time()
            response = self.client.automatic_speech_recognition(
                audio_path, 
                model=Config.MODEL_STT
            )
            transcription = response.text.strip()
            return transcription, (time.time() - start_t)
        except Exception:
            try:
                response = self.client.automatic_speech_recognition(
                    audio_path, 
                    model="openai/whisper-small"
                )
                return response.text.strip(), 0.0
            except Exception as e:
                return None, 0.0

    async def _generate_speech_edge(self, text, output_file):
        """Generate high-quality speech using Edge TTS"""
        communicate = edge_tts.Communicate(text, Config.VOICE_MALE)
        await communicate.save(output_file)

    def _generate_speech_gtts(self, text, output_file):
        """Fallback speech generation using gTTS"""
        tts = gTTS(text=text, lang='en', slow=False, tld='com')
        tts.save(output_file)

    def speak(self, text):
        """Convert text to speech with dual-engine fallback"""
        start_t = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name

            # Try Edge TTS first (better quality)
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
                
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                    os.unlink(tmp_path)
                    return audio_bytes, (time.time() - start_t)
            except:
                pass

            # Fallback to gTTS
            self._generate_speech_gtts(text, tmp_path)
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            os.unlink(tmp_path)
            
            return audio_bytes, (time.time() - start_t)
            
        except Exception:
            return None, 0.0

# ==============================================================================
# ENHANCED BRAIN ENGINE
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
        """Generate contextually accurate response as Mahesh"""
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
                max_tokens=250,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            return answer, (time.time() - start_t)
            
        except Exception:
            # Fallback to backup models
            backups = ["mistralai/Mistral-7B-Instruct-v0.2", "HuggingFaceH4/zephyr-7b-beta"]
            for backup in backups:
                try:
                    response = self.client.chat_completion(
                        messages=[
                            {"role": "system", "content": MAHESH_PERSONA},
                            {"role": "user", "content": question}
                        ],
                        model=backup,
                        max_tokens=250
                    )
                    return response.choices[0].message.content.strip(), 0.0
                except:
                    continue
            return None, 0.0

# ==============================================================================
# PROFESSIONAL STREAMLIT UI
# ==============================================================================

def main():
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Professional CSS Styling
    st.markdown("""
        <style>
        /* Main Container */
        .main {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            padding: 0;
        }
        
        /* Header */
        .pro-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .pro-header h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
        }
        
        .pro-header p {
            margin: 10px 0 0 0;
            font-size: 1.1rem;
            opacity: 0.95;
        }
        
        /* Message Bubbles */
        .user-message {
            background: linear-gradient(135deg, #4e54c8 0%, #8f94fb 100%);
            color: white;
            padding: 20px;
            border-radius: 20px 20px 5px 20px;
            margin: 15px 0;
            box-shadow: 0 5px 15px rgba(78, 84, 200, 0.3);
            animation: slideInRight 0.3s ease-out;
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 20px;
            border-radius: 20px 20px 20px 5px;
            margin: 15px 0;
            box-shadow: 0 5px 15px rgba(17, 153, 142, 0.3);
            animation: slideInLeft 0.3s ease-out;
        }
        
        .message-label {
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 8px;
            opacity: 0.9;
        }
        
        .message-content {
            font-size: 1.05rem;
            line-height: 1.6;
        }
        
        /* Info Boxes */
        .info-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            color: white;
        }
        
        .info-card h3 {
            color: #8f94fb;
            margin-top: 0;
        }
        
        .info-card ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        
        .info-card li {
            margin: 8px 0;
            line-height: 1.5;
        }
        
        /* Status Messages */
        .status-box {
            background: rgba(17, 153, 142, 0.2);
            border-left: 4px solid #38ef7d;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            color: white;
        }
        
        .error-box {
            background: rgba(231, 76, 60, 0.2);
            border-left: 4px solid #e74c3c;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            color: white;
        }
        
        /* Animations */
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
            .pro-header h1 {
                font-size: 1.8rem;
            }
            .pro-header p {
                font-size: 1rem;
            }
            .user-message, .assistant-message {
                padding: 15px;
            }
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # Initialize Session State
    if "brain" not in st.session_state:
        st.session_state.brain = BrainEngine()
        st.session_state.audio = AudioEngine()
        st.session_state.conversation = []

    # Header
    st.markdown("""
        <div class='pro-header'>
            <h1>🎙️ Mahesh AI Voice Agent</h1>
            <p>Professional Interview Bot | Stage 1 Submission</p>
        </div>
    """, unsafe_allow_html=True)

    # Show example questions if first time
    if len(st.session_state.conversation) == 0:
        st.markdown("""
        <div class='info-card'>
            <h3>📝 Ask Me About:</h3>
            <ul>
                <li><strong>My Life Story</strong> - Journey from Mechanical to AI</li>
                <li><strong>My Superpower</strong> - Systematic problem-solving</li>
                <li><strong>Growth Areas</strong> - Agentic AI, Cloud, Communication</li>
                <li><strong>Misconceptions</strong> - Software skills perception</li>
                <li><strong>Pushing Limits</strong> - Weekly prototypes & learning</li>
            </ul>
            <p style='margin-top: 15px; font-size: 0.95rem; opacity: 0.8;'>
                💡 Click the microphone below and ask any question naturally
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Conversation History
    if st.session_state.conversation:
        st.markdown("---")
        for msg in st.session_state.conversation:
            if msg['role'] == 'user':
                st.markdown(f"""
                    <div class='user-message'>
                        <div class='message-label'>👤 YOU ASKED:</div>
                        <div class='message-content'>{msg['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='assistant-message'>
                        <div class='message-label'>🎙️ MAHESH:</div>
                        <div class='message-content'>{msg['content']}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                if msg.get('audio'):
                    st.audio(msg['audio'], format="audio/mp3")
        
        st.markdown("---")
        
        if st.button("🔄 Start New Conversation", use_container_width=True):
            st.session_state.conversation = []
            st.rerun()

    # Voice Input Section
    st.markdown("### 🎤 Record Your Question")
    audio_input = st.audio_input("Click to start recording")

    if audio_input:
        # Process audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_input.getvalue())
            tmp_path = tmp.name

        # Step 1: Speech Recognition
        with st.spinner("🎧 Listening..."):
            question, listen_time = st.session_state.audio.listen(tmp_path)
            os.unlink(tmp_path)
        
        if not question:
            st.markdown("""
                <div class='error-box'>
                    ❌ <strong>Could not understand audio.</strong> Please try again with clear speech.
                </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Display recognized question
        st.markdown(f"""
            <div class='status-box'>
                ✅ <strong>I heard:</strong> "{question}"
            </div>
        """, unsafe_allow_html=True)

        # Step 2: Generate Response
        with st.spinner("🧠 Thinking..."):
            answer, think_time = st.session_state.brain.think(question)
        
        if not answer:
            st.markdown("""
                <div class='error-box'>
                    ❌ <strong>Could not generate response.</strong> Please try again.
                </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Step 3: Text to Speech
        with st.spinner("🗣️ Generating voice..."):
            audio_bytes, speak_time = st.session_state.audio.speak(answer)

        # Display Result
        st.markdown(f"""
            <div class='user-message'>
                <div class='message-label'>👤 YOU ASKED:</div>
                <div class='message-content'>{question}</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='assistant-message'>
                <div class='message-label'>🎙️ MAHESH:</div>
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
            
            # Performance metrics in expander
            with st.expander("⚡ Performance Details"):
                col1, col2, col3 = st.columns(3)
                col1.metric("🎧 Listening", f"{listen_time:.1f}s")
                col2.metric("🧠 Processing", f"{think_time:.1f}s")
                col3.metric("🗣️ Speaking", f"{speak_time:.1f}s")
        else:
            st.markdown("""
                <div class='error-box'>
                    ⚠️ <strong>Voice generation failed.</strong> Text response shown above.
                </div>
            """, unsafe_allow_html=True)

    # Footer Info
    if len(st.session_state.conversation) == 0:
        st.markdown("""
        <div class='info-card' style='margin-top: 30px;'>
            <h3>ℹ️ How It Works</h3>
            <p><strong>1. Record:</strong> Click the microphone and ask your question</p>
            <p><strong>2. Process:</strong> AI transcribes, thinks, and responds</p>
            <p><strong>3. Listen:</strong> Hear Mahesh's voice answer</p>
            <p style='margin-top: 15px; font-size: 0.9rem; opacity: 0.7;'>
                🔧 Powered by: Whisper STT • HuggingFace LLM • Edge TTS
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
