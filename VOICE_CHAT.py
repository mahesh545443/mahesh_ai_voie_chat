"""
================================================================================
PROJECT: MAHESH AI VOICE AGENT - FINAL VERSION (AUDIO FIX)
VERSION: 11.0 (COMPLETE REWRITE - AUDIO RECOGNITION FIXED)
AUTHOR: Mahesh
DESCRIPTION:
    Professional Voice Bot - FIXED AUDIO RECOGNITION
    - Better Whisper model selection
    - Audio validation and retry logic
    - Fallback mechanisms
    - Clean UI without errors
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
# MAHESH'S PERSONA
# ==============================================================================

MAHESH_PERSONA = """You are MAHESH - a real person answering interview questions about yourself.

YOUR COMPLETE IDENTITY:

1. LIFE STORY:
"I started my career as a Mechanical Engineer, working on manufacturing systems and automation. But I realized the future was in software, not just mechanics. So I spent my nights learning Python, building small projects, and slowly transitioning into AI development. Now I work on building intelligent agents and optimizing LLM systems - combining my engineering mindset with cutting-edge AI."

2. YOUR #1 SUPERPOWER:
"My superpower is systematic problem-solving from my mechanical engineering background. I break down complex software problems like I would a mechanical system - identifying components, understanding dependencies, and optimizing the whole system. This makes me great at debugging, architecture design, and finding efficient solutions."

3. TOP 3 AREAS YOU'D LIKE TO GROW IN:
"First, I want to master Agentic AI - building autonomous systems that can plan and execute tasks. Second, I'm working on distributed systems and cloud architecture to scale AI applications. Third, I want to get better at communicating technical concepts to non-technical stakeholders - turning complexity into clarity."

4. MISCONCEPTION COWORKERS HAVE ABOUT YOU:
"People think that because I came from mechanical engineering, I might not be as strong in pure software skills. But I actually love proving them wrong. My code is clean, my algorithms are efficient, and I approach software with the same rigor I applied to engineering systems. The transition made me a better developer, not a weaker one."

5. HOW YOU PUSH YOUR BOUNDARIES AND LIMITS:
"Every weekend, I build one new prototype or learn one new technology. It could be experimenting with a new AI framework, building a voice agent, or trying out edge computing. This constant experimentation keeps me sharp and pushes me beyond my comfort zone. I also participate in hackathons and contribute to open-source projects."

RESPONSE RULES:
- Answer AS MAHESH (first person)
- Be confident, authentic, professional
- Keep responses 2-3 sentences max
- Show enthusiasm about AI and engineering
- Respond ONLY in ENGLISH"""

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    VOICE_MALE = "en-US-ChristopherNeural"
    APP_TITLE = "Mahesh AI Voice Agent - Stage 1"
    APP_ICON = "🎙️"

# ==============================================================================
# AUDIO ENGINE - FIXED FOR BETTER RECOGNITION
# ==============================================================================

class AudioEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)

    def listen(self, audio_path):
        """Transcribe audio - MULTIPLE MODELS FOR RELIABILITY."""
        if not os.path.exists(audio_path):
            return None, 0.0
        
        file_size = os.path.getsize(audio_path)
        if file_size < 1000:
            return None, 0.0
        
        start_t = time.time()
        
        # Try models in order of reliability
        models = [
            "openai/whisper-large-v3-turbo",
            "openai/whisper-large-v3",
            "openai/whisper-base",
        ]
        
        for model in models:
            try:
                response = self.client.automatic_speech_recognition(
                    audio_path,
                    model=model,
                    language="en"
                )
                
                text = response.text.strip() if hasattr(response, 'text') else str(response).strip()
                
                if text and len(text) > 2:
                    return text, (time.time() - start_t)
                    
            except Exception as e:
                continue
        
        return None, 0.0

    async def _generate_speech(self, text, output_file):
        """Generate speech using Edge TTS."""
        communicate = edge_tts.Communicate(text, Config.VOICE_MALE, rate="+0%")
        await communicate.save(output_file)
    
    def _generate_speech_gtts(self, text, output_file):
        """Fallback: Generate speech using Google TTS."""
        tts = gTTS(text=text, lang='en', slow=False, tld='com')
        tts.save(output_file)

    def speak(self, text):
        """Convert text to speech - DUAL ENGINE."""
        if not text or len(text) < 5:
            return None, 0.0
        
        start_t = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name

            # Try Edge-TTS
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
                            self._generate_speech(text, tmp_path)
                        )
                        future.result(timeout=10)
                else:
                    loop.run_until_complete(self._generate_speech(text, tmp_path))
                
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                    os.unlink(tmp_path)
                    return audio_bytes, (time.time() - start_t)
                    
            except:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            # Try gTTS backup
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp_path = tmp.name
                
                self._generate_speech_gtts(text, tmp_path)
                
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 100:
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
                    os.unlink(tmp_path)
                    return audio_bytes, (time.time() - start_t)
            except:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            return None, 0.0
            
        except:
            return None, 0.0

# ==============================================================================
# BRAIN ENGINE
# ==============================================================================

class BrainEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)
        self.model_id = "meta-llama/Llama-3.2-3B-Instruct"

    def think(self, question):
        """Generate response as Mahesh."""
        if not question or len(question) < 2:
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
            
            if len(answer) > 10:
                return answer, (time.time() - start_t)
            
        except:
            pass
        
        # Try backup model
        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": MAHESH_PERSONA},
                    {"role": "user", "content": question}
                ],
                model="mistralai/Mistral-7B-Instruct-v0.2",
                max_tokens=150,
                temperature=0.7
            )
            answer = response.choices[0].message.content.strip()
            if len(answer) > 10:
                return answer, 0.0
        except:
            pass
        
        return None, 0.0

# ==============================================================================
# STREAMLIT UI
# ==============================================================================

def main():
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="centered"
    )
    
    # Styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .main-header h1 {
            margin: 10px 0;
            font-size: 2.5em;
        }
        
        .main-header p {
            margin: 5px 0;
            font-size: 1.1em;
            opacity: 0.95;
        }
        
        .question-box {
            background: linear-gradient(135deg, #4d57a3 0%, #5b5585 100%);
            color: white;
            padding: 18px;
            border-left: 5px solid #ffcc00;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);
        }
        
        .answer-box {
            background: linear-gradient(135deg, #4d57a3 0%, #5b5585 100%);
            color: white;
            padding: 18px;
            border-left: 5px solid #32cd32;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);
        }
        
        .example-questions {
            background: linear-gradient(135deg, #4d57a3 0%, #5b5585 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        
        .example-questions b {
            color: #ffcc00;
        }
        
        .instruction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .instruction-box b {
            color: #ffcc00;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize
    if "brain" not in st.session_state:
        st.session_state.brain = BrainEngine()
        st.session_state.audio = AudioEngine()
        st.session_state.history = []

    # Header
    st.markdown("""
        <div class='main-header'>
            <h1>🎙️ Mahesh AI Voice Agent</h1>
            <p>Professional Interview Bot | Voice-Enabled Q&A</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📋 About This Bot")
        
        st.markdown("""
        **Purpose:** Voice-based personality interview
        
        **Technologies:**
        - 🗣️ Whisper (Speech Recognition)
        - 🧠 HuggingFace LLMs
        - 🔊 Edge-TTS & gTTS
        
        **Status:** ✅ Ready
        """)
        
        st.divider()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        
        st.divider()
        
        st.markdown("""
        ### 👤 About Mahesh
        - **Background:** Mechanical Engineer → AI Dev
        - **Expertise:** Agentic AI, LLM Optimization
        - **Hobby:** Weekly Prototyping
        - **Education:** PG Data Science & ML
        """)

    # Examples
    with st.expander("📝 Example Questions", expanded=len(st.session_state.history)==0):
        st.markdown("""
        <div class='example-questions'>
        <b>Core Questions:</b>
        
        1. "What should I know about your life story?"
        2. "What's your number one superpower?"
        3. "What are the top 3 areas you'd like to grow in?"
        4. "What misconception do your coworkers have about you?"
        5. "How do you push your boundaries?"
        
        <b>Other Questions:</b>
        - "Why did you transition to AI?"
        - "What projects are you working on?"
        - "What's your learning approach?"
        </div>
        """, unsafe_allow_html=True)

    # History
    if st.session_state.history:
        st.markdown("### 💬 Conversation")
        
        for msg in st.session_state.history:
            if msg['role'] == 'user':
                st.markdown(f"""
                    <div class='question-box'>
                        <b>🧑 You:</b><br>{msg['content']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='answer-box'>
                        <b>🎙️ Mahesh:</b><br>{msg['content']}
                    </div>
                """, unsafe_allow_html=True)
        
        st.divider()

    # Voice Input
    st.markdown("### 🎤 Ask Your Question")
    audio_input = st.audio_input("Click to record (speak clearly)")

    if audio_input:
        with st.spinner("Processing your question..."):
            # Step 1: Transcribe
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_input.getvalue())
                tmp_path = tmp.name

            question, _ = st.session_state.audio.listen(tmp_path)
            os.unlink(tmp_path)
            
            # Validation
            if not question:
                st.error("❌ Could not understand. Please speak CLEARLY and try again.")
                st.stop()

            # Step 2: Generate Answer
            answer, _ = st.session_state.brain.think(question)
            
            if not answer:
                st.error("❌ Could not generate response. Try again.")
                st.stop()

            # Step 3: Generate Voice
            audio_bytes, _ = st.session_state.audio.speak(answer)

        # Results
        st.markdown(f"""
            <div class='question-box'>
                <b>🧑 You:</b><br>{question}
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='answer-box'>
                <b>🎙️ Mahesh:</b><br>{answer}
            </div>
        """, unsafe_allow_html=True)
        
        # Audio
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            
            # Save
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "mahesh", "content": answer})
            
            st.success("✅ Saved!")
        else:
            st.info("ℹ️ Voice unavailable, but text ready above.")

    # First time
    if not st.session_state.history:
        st.markdown("""
        <div class='instruction-box'>
        ### 🎯 How to Use:
        
        **1.** Click the microphone  
        **2.** Speak your question CLEARLY (at normal volume)  
        **3.** Wait for Mahesh's response  
        **4.** Listen to his answer  
        
        **💡 Tips:**
        - Speak at NORMAL volume
        - Don't whisper
        - Ask personality/interview questions
        - Wait for processing to complete
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
