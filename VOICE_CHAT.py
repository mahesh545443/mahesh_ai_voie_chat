"""
================================================================================
PROJECT: MAHESH AI VOICE AGENT - OPTIMIZED STAGE 1 SUBMISSION
VERSION: 9.5 (Cleaned, Faster, Professional UI)
AUTHOR: Mahesh & Gemini
DESCRIPTION:
    Voice bot that answers personality questions AS MAHESH using:
    - HuggingFace Chat API
    - Whisper for Speech-to-Text
    - Edge-TTS for high-quality Text-to-Speech (with gTTS fallback)
    - NEW: Streamlit Native UI (st.chat_message) for better mobile experience
    - NEW: Added Text Input (st.chat_input)
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

# 🔊 CRITICAL FIX: Apply nest_asyncio at module level for Edge-TTS
nest_asyncio.apply()

# ==============================================================================
# MAHESH'S PERSONA DATABASE (The LLM's System Prompt)
# ==============================================================================

MAHESH_PERSONA = """You are MAHESH - a real person answering interview questions about yourself.

YOUR COMPLETE IDENTITY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. LIFE STORY (in a few sentences):
"I started my career as a Mechanical Engineer, working on manufacturing systems and automation. But I realized the future was in software, not just mechanics. So I spent my nights learning Python, building small projects, and slowly transitioning into AI development. Now I work on building intelligent agents and optimizing LLM systems - combining my engineering mindset with cutting-edge AI."

2. YOUR #1 SUPERPOWER:
"My superpower is systematic problem-solving from my mechanical engineering background. I break down complex software problems like I would a mechanical system - identifying components, understanding dependencies, and optimizing the whole system. This makes me great at debugging, architecture design, and finding efficient solutions."

3. TOP 3 AREAS YOU'D LIKE TO GROW IN:
"First, I want to master Agentic AI - building autonomous systems that can plan and execute tasks. Second, I'm working on distributed systems and cloud architecture to scale AI applications. Third, I want to get better at communicating technical concepts to non-technical stakeholders - turning complexity into clarity."

4. MISCONCEPTION COWORKERS HAVE ABOUT YOU:
"People think that because I came from mechanical engineering, I might not be as strong in pure software skills. But I actually love proving them wrong. My code is clean, my algorithms are efficient, and I approach software with the same rigor I applied to engineering systems. The transition made me a better developer, not a weaker one."

5. HOW YOU PUSH YOUR BOUNDARIES AND LIMITS:
"Every weekend, I build one new prototype or learn one new technology. It could be experimenting with a new AI framework, building a voice agent, or trying out edge computing. This constant experimentation keeps me sharp and pushes me beyond my comfort zone. I also participate in hackathons and contribute to open-source projects."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESPONSE RULES:
- Answer AS MAHESH (first person: "I", "my", "me")
- Be confident, authentic, and professional
- Keep responses concise (2-4 sentences)
- Reference specific details from your story above
- Show personality - you're enthusiastic about AI and engineering
- If asked variations of the 5 questions, use the exact answers above
- For other questions, stay consistent with this persona"""

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    HF_TOKEN = st.secrets["HF_TOKEN"]
    # Single, reliable models for stability
    MODEL_STT = "openai/whisper-large-v3-turbo" 
    MODEL_LLM = "mistralai/Mistral-7B-Instruct-v0.2" # Excellent and fast
    VOICE_MALE = "en-US-ChristopherNeural"
    
    APP_TITLE = "Mahesh AI Voice Agent"
    APP_ICON = "🎙️"

# ==============================================================================
# AUDIO ENGINE (Edge-TTS with gTTS fallback)
# ==============================================================================

class AudioEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)

    @st.cache_data(show_spinner=False, max_entries=100)
    def listen(_self, audio_bytes):
        """Transcribe audio using Whisper."""
        try:
            start_t = time.time()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes.getvalue())
                tmp_path = tmp.name
            
            response = _self.client.automatic_speech_recognition(
                tmp_path, model=Config.MODEL_STT
            )
            os.unlink(tmp_path)
            return response.text, (time.time() - start_t)
        except Exception as e:
            return f"[ERROR] STT Failed: {str(e)[:50]}...", 0.0

    async def _generate_speech_edge(self, text, output_file):
        """Generate speech using Edge TTS (Best Quality)."""
        communicate = edge_tts.Communicate(text, Config.VOICE_MALE)
        await communicate.save(output_file)
    
    def _generate_speech_gtts(self, text, output_file):
        """Fallback: Generate speech using Google TTS (Always Works)."""
        tts = gTTS(text=text, lang='en', slow=False, tld='com')
        tts.save(output_file)

    def speak(self, text):
        """Convert text to speech - Tries Edge-TTS, falls back to gTTS."""
        if "[ERROR]" in text: 
            return None, 0.0
        
        start_t = time.time()
        audio_bytes = None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name
            
            # 🔊 TRY METHOD 1: Edge-TTS (Best Quality)
            try:
                # Use a cleaner way to run the async Edge-TTS function
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self._generate_speech_edge(text, tmp_path))
                
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
                    with open(tmp_path, "rb") as f:
                        audio_bytes = f.read()
            except Exception:
                # 🔊 METHOD 2: gTTS Fallback
                self._generate_speech_gtts(text, tmp_path)
                with open(tmp_path, "rb") as f:
                    audio_bytes = f.read()

            if audio_bytes and len(audio_bytes) > 100:
                return audio_bytes, (time.time() - start_t)
            else:
                raise Exception("Generated audio file is too small.")
            
        except Exception as e:
            st.warning(f"🔊 Both voice engines failed. Using text only.")
            return None, 0.0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

# ==============================================================================
# BRAIN ENGINE (HUGGINGFACE CHAT API)
# ==============================================================================

class BrainEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)

    @st.cache_data(show_spinner=False, max_entries=100)
    def think(_self, question):
        """Generate response as Mahesh using the specified persona."""
        try:
            start_t = time.time()
            
            messages = [
                {"role": "system", "content": MAHESH_PERSONA},
                {"role": "user", "content": question}
            ]
            
            response = _self.client.chat_completion(
                messages=messages,
                model=Config.MODEL_LLM,
                max_tokens=200,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            return answer, (time.time() - start_t)
            
        except Exception as e:
            st.error(f"LLM Error: Could not connect to {Config.MODEL_LLM}. {str(e)[:50]}")
            return "I'm having trouble connecting to the network right now. Please try again.", 0.0

# ==============================================================================
# STREAMLIT UI & EXECUTION FLOW
# ==============================================================================

def execute_chat_flow(user_input, is_voice=False):
    """Handles the full flow: Think -> Speak -> Display."""
    
    # 1. Generate Answer
    with st.spinner("🧠 Mahesh is thinking..."):
        answer, think_time = st.session_state.brain.think(user_input)
    
    # 2. Generate Voice
    with st.spinner("🔊 Generating voice..."):
        audio_bytes, speak_time = st.session_state.audio.speak(answer)
    
    # 3. Save History
    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "mahesh", "content": answer, "audio": audio_bytes})
    
    # 4. Rerun to display new history
    st.rerun()

def process_voice_input(audio_input):
    """Handles the Voice-to-Text flow."""
    with st.spinner("👂 Listening and Transcribing..."):
        question, listen_time = st.session_state.audio.listen(audio_input)
    
    if "[ERROR]" in question:
        st.error(f"Transcription failed: {question}")
        return
    
    # Pass the transcribed question to the main chat flow
    execute_chat_flow(question, is_voice=True)

def display_chat_history():
    """Renders all messages using native st.chat_message for cleaner UI."""
    for msg in st.session_state.history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🎙️"):
                st.write(msg["content"])
                if msg.get("audio"):
                    # Use a clean container for the audio player
                    st.audio(msg["audio"], format="audio/mp3", autoplay=True)

def main():
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="centered"
    )
    
    st.title(f"{Config.APP_ICON} {Config.APP_TITLE}")
    st.caption("Stage 1 Interview Submission | Voice-Enabled Q&A Bot")
    st.divider()

    # Initialize Engines
    if "brain" not in st.session_state:
        with st.spinner("🚀 Initializing Mahesh's Brain and Audio Engines..."):
            st.session_state.brain = BrainEngine()
            st.session_state.audio = AudioEngine()
            st.session_state.history = []
            st.success(f"✅ System Ready using {Config.MODEL_LLM.split('/')[-1]}!")
            time.sleep(0.5)
            st.rerun() # Clear spinner cleanly

    # --- UI LAYOUT ---

    # 1. Conversation History Display
    if st.session_state.history:
        display_chat_history()
    else:
        # Initial instructions only shown on first run
        st.info("""
        **Welcome!** Ask Mahesh a question using **voice (microphone)** or **text (chatbox)** below.
        
        🗣️ **Try asking:** "What's your number one superpower?"
        """)
    
    st.divider()

    # 2. Input Method Container
    col_text, col_voice = st.columns([3, 1])

    # A. Text Input (st.chat_input)
    with col_text:
        text_input = st.chat_input("Type your question here...")
        if text_input:
            execute_chat_flow(text_input)

    # B. Voice Input (st.audio_input)
    with col_voice:
        # Use a button to record, then handle the resulting audio file
        audio_input = st.audio_recorder(
            label="Record Voice Question", 
            icon="🎙️",
            text="Record",
            recording_color="#ea7466",
            icon_size="1x"
        )
        if audio_input:
            process_voice_input(audio_input)

    # 3. Sidebar
    with st.sidebar:
        st.title("⚙️ Debug & Controls")
        
        st.markdown(f"**LLM Model:** `{Config.MODEL_LLM.split('/')[-1]}`")
        st.markdown(f"**TTS Voice:** `{Config.VOICE_MALE}`")
        
        st.divider()
        
        if st.button("🔄 Clear Conversation"):
            st.session_state.history = []
            st.toast("Conversation cleared!")
            st.rerun()

if __name__ == "__main__":
    main()

