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
# CONFIGURATION
# ==============================================================================

class Config:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    MODEL_STT = "openai/whisper-large-v3-turbo"
    VOICE_MALE = "en-IN-PrabhatNeural"  # Indian male voice - deep and professional
    APP_TITLE = "Mahesh AI Voice Agent"
    APP_ICON = "🎙️"

# ==============================================================================
# ENHANCED AUDIO ENGINE
# ==============================================================================

class AudioEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)

    def listen(self, audio_path):
        """Enhanced speech recognition with noise filtering"""
        try:
            start_t = time.time()
            # Use Whisper with better parameters for noisy environments
            response = self.client.automatic_speech_recognition(
                audio_path, 
                model=Config.MODEL_STT
            )
            transcription = response.text.strip()
            
            # Filter out very short or empty responses
            if len(transcription) < 3:
                return None, 0.0
                
            return transcription, (time.time() - start_t)
        except Exception:
            try:
                # Fallback to smaller model
                response = self.client.automatic_speech_recognition(
                    audio_path, 
                    model="openai/whisper-medium"
                )
                transcription = response.text.strip()
                if len(transcription) < 3:
                    return None, 0.0
                return transcription, 0.0
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
        """Generate contextually accurate response as Mahesh - OPTIMIZED FOR SPEED"""
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
                max_tokens=150,  # Reduced for faster response
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            return answer, (time.time() - start_t)
            
        except Exception:
            # Quick fallback
            return "I'm having trouble connecting. Please try again.", 0.0

# ==============================================================================
# PROFESSIONAL STREAMLIT UI - DARK THEME
# ==============================================================================

def main():
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # DARK THEME PROFESSIONAL CSS
    st.markdown("""
        <style>
        /* Force Dark Background for Everything */
        .main, .stApp, [data-testid="stAppViewContainer"], 
        [data-testid="stHeader"], section[data-testid="stSidebar"] {
            background-color: #0a0e27 !important;
            color: #e0e0e0 !important;
        }
        
        /* Header Section */
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
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        }
        
        .pro-header p {
            margin: 12px 0 0 0;
            font-size: 1.1rem;
            color: #f0f0f0;
            opacity: 0.95;
        }
        
        /* User Message Bubble */
        .user-message {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: #ffffff;
            padding: 20px 24px;
            border-radius: 18px 18px 4px 18px;
            margin: 20px 0;
            box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
            animation: slideInRight 0.4s ease-out;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
        
        /* Assistant Message Bubble */
        .assistant-message {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: #ffffff;
            padding: 20px 24px;
            border-radius: 18px 18px 18px 4px;
            margin: 20px 0;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
            animation: slideInLeft 0.4s ease-out;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .message-label {
            font-weight: 700;
            font-size: 0.85rem;
            margin-bottom: 10px;
            opacity: 0.9;
            color: #ffffff;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .message-content {
            font-size: 1.08rem;
            line-height: 1.7;
            color: #ffffff;
        }
        
        /* Info Card - Dark Theme */
        .info-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border: 1px solid #475569;
            border-radius: 16px;
            padding: 28px;
            margin: 25px 0;
            color: #e2e8f0;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }
        
        .info-card h3 {
            color: #a78bfa;
            margin-top: 0;
            font-size: 1.4rem;
            font-weight: 700;
        }
        
        .info-card ul {
            margin: 15px 0;
            padding-left: 24px;
        }
        
        .info-card li {
            margin: 10px 0;
            line-height: 1.6;
            color: #cbd5e1;
        }
        
        .info-card li strong {
            color: #f0abfc;
        }
        
        .info-card p {
            color: #cbd5e1;
        }
        
        /* Status Box - Success */
        .status-box {
            background: linear-gradient(135deg, #065f46 0%, #047857 100%);
            border-left: 5px solid #10b981;
            padding: 18px 20px;
            border-radius: 10px;
            margin: 18px 0;
            color: #ffffff;
            box-shadow: 0 3px 12px rgba(16, 185, 129, 0.3);
        }
        
        .status-box strong {
            color: #d1fae5;
        }
        
        /* Error Box */
        .error-box {
            background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
            border-left: 5px solid #ef4444;
            padding: 18px 20px;
            border-radius: 10px;
            margin: 18px 0;
            color: #ffffff;
            box-shadow: 0 3px 12px rgba(239, 68, 68, 0.3);
        }
        
        .error-box strong {
            color: #fecaca;
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
            from {
                opacity: 0;
                transform: translateX(40px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-40px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Streamlit Component Styling */
        .stAudioInput, .stButton button {
            border-radius: 12px !important;
        }
        
        .stButton button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 12px 24px;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
            transform: translateY(-2px);
        }
        
        /* New Chat Button - ChatGPT Style */
        .new-chat-button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            margin: 20px 0;
            cursor: pointer;
            border: none;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            transition: all 0.3s ease;
        }
        
        .new-chat-button:hover {
            background: linear-gradient(135deg, #059669 0%, #047857 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
        }
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
            .pro-header h1 {
                font-size: 1.9rem;
            }
            .pro-header p {
                font-size: 1rem;
            }
            .user-message, .assistant-message {
                padding: 16px 18px;
            }
            .info-card {
                padding: 20px;
            }
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: #1e293b !important;
            color: #e2e8f0 !important;
            border-radius: 8px !important;
        }
        
        /* Audio Player Styling */
        audio {
            width: 100%;
            border-radius: 8px;
        }
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
            <h1>🎙️ Mahesh - AI Engineer</h1>
            <p>Voice-Enabled Interview Assistant | Professional Q&A Bot</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar with Clear History and Info
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        # New Chat Button in Sidebar (ChatGPT Style)
        if st.button("➕ New Chat", use_container_width=True, key="sidebar_new_chat"):
            st.session_state.conversation = []
            st.rerun()
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.conversation = []
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 👤 About Mahesh")
        st.markdown("""
        **AI Engineer** (2.5+ years exp)
        
        🎓 B.Tech Mechanical Engineering  
        🎓 PG in Data Science & Advanced ML  
        💼 Currently: AI Engineer  
        🚀 Specializes in Agentic AI & LLMs  
        """)
        
        st.markdown("---")
        
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        - **Speech-to-Text**: Whisper Large V3
        - **AI Brain**: HuggingFace LLM
        - **Text-to-Speech**: Edge TTS (Male Voice)
        """)
        
        st.markdown("---")
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Speak in a quiet place
        - Ask questions clearly
        - One question at a time
        """)


    # Show example questions if first time (in expandable section)
    if len(st.session_state.conversation) == 0:
        with st.expander("📝 **Ask Me About** (Click to expand)", expanded=False):
            st.markdown("""
            **Sample Questions You Can Ask:**
            
            - **My Life Story** - Journey from Mechanical Engineering to AI Development
            - **My #1 Superpower** - Systematic problem-solving approach
            - **Top 3 Growth Areas** - Agentic AI, Cloud Architecture, Communication
            - **Misconceptions** - What coworkers misunderstand about me
            - **Pushing Boundaries** - How I challenge myself weekly
            
            💡 **Tip**: Speak clearly in a quiet environment for best results
            """)
        
        with st.expander("ℹ️ **How It Works** (Click to expand)", expanded=False):
            st.markdown("""
            **Step 1:** Click the microphone and record your question  
            **Step 2:** AI transcribes your voice and generates response  
            **Step 3:** Listen to Mahesh's natural voice answer  
            
            🔧 **Technology**: Whisper STT • HuggingFace LLM • Edge TTS
            
            ⚠️ **For Best Results:**
            - Speak in a quiet environment
            - Speak clearly and at normal pace
            - Ask one question at a time
            """)
    else:
        # Show dropdowns even after conversation starts
        with st.expander("📝 **Ask Me About**", expanded=False):
            st.markdown("""
            **Sample Questions:**
            
            - My Life Story - Journey from Mechanical Engineering to AI Development
            - My #1 Superpower - Systematic problem-solving approach
            - Top 3 Growth Areas - Agentic AI, Cloud Architecture, Communication
            - Misconceptions - What coworkers misunderstand about me
            - Pushing Boundaries - How I challenge myself weekly
            """)
        
        with st.expander("ℹ️ **How It Works**", expanded=False):
            st.markdown("""
            **Step 1:** Record your question  
            **Step 2:** AI processes and responds  
            **Step 3:** Listen to the voice response  
            
            🔧 **Tech**: Whisper STT • HuggingFace LLM • Edge TTS
            """)

    # Conversation History with New Chat Button (ChatGPT Style)
    if st.session_state.conversation:
        st.markdown("<div class='section-title'>💬 Conversation History</div>", unsafe_allow_html=True)
        
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
    audio_input = st.audio_input("Click to start recording your question")

    if audio_input:
        # Process audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_input.getvalue())
            tmp_path = tmp.name

        # Step 1: Speech Recognition
        with st.spinner("🎧 Listening to your question..."):
            question, listen_time = st.session_state.audio.listen(tmp_path)
            os.unlink(tmp_path)
        
        if not question:
            st.markdown("""
                <div class='error-box'>
                    ❌ <strong>Could not understand audio.</strong><br>
                    💡 <strong>Tips for better recognition:</strong><br>
                    • Find a quiet place with minimal background noise<br>
                    • Speak clearly at a normal pace<br>
                    • Hold the device closer to your mouth<br>
                    • Avoid recording in echoey rooms
                </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Display recognized question
        st.markdown(f"""
            <div class='status-box'>
                ✅ <strong>I heard you say:</strong> "{question}"
            </div>
        """, unsafe_allow_html=True)

        # Step 2: Generate Response
        with st.spinner("🧠 Thinking and generating response..."):
            answer, think_time = st.session_state.brain.think(question)
        
        if not answer:
            st.markdown("""
                <div class='error-box'>
                    ❌ <strong>Could not generate response.</strong> Please try again.
                </div>
            """, unsafe_allow_html=True)
            st.stop()

        # Step 3: Text to Speech
        with st.spinner("🗣️ Converting to voice..."):
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
                <div class='error-box'>
                    ⚠️ <strong>Voice generation failed.</strong> Text response is shown above.
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
       
