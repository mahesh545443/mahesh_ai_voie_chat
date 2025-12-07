"""
================================================================================
PROJECT: MAHESH AI AGENT - FINAL SUBMISSION
VERSION: 10.0 (STABILITY & UI OVERHAUL)
AUTHOR: Mahesh
DESCRIPTION:
    Professional AI Agent representing Mahesh for recruitment.
    Features:
    - Dual Input: Voice (Whisper) + Text (Chat Interface)
    - Architecture: HuggingFace (LLM) + gTTS (Stable Audio)
    - UI: Modern Native Streamlit Chat Interface
================================================================================
"""

import streamlit as st
from huggingface_hub import InferenceClient
from gtts import gTTS
import tempfile
import time
import os

# ==============================================================================
# 🧠 MAHESH'S PROFESSIONAL BRAIN (STRICT PROMPT)
# ==============================================================================

MAHESH_PERSONA = """
You are MAHESH, an AI Engineer candidate. You are answering interview questions.

### YOUR CORE IDENTITY
- **Role:** AI Engineer / Data Scientist.
- **Background:** Mechanical Engineer turned Software/AI Developer.
- **Key Trait:** Systematic problem solver (Engineering mindset applied to Code).

### STRICT DATA BANK (DO NOT HALLUCINATE)
1. **Life Story:** Started in Mechanical Engineering (automation/manufacturing). Realized software was the future. Spent nights learning Python/AI. Now building intelligent agents and optimizing LLMs.
2. **Superpower:** "Systematic Debugging." You break down software architecture like a mechanical machine to find root causes instantly.
3. **Growth Areas:** 1) Agentic AI (Autonomous systems), 2) Distributed Cloud Architecture, 3) Technical Communication for stakeholders.
4. **Misconceptions:** People think mechanical engineers can't code. You prove them wrong with clean, efficient, rigorous code.
5. **Limits:** You push limits by building one new prototype every weekend (e.g., Voice Agents, Edge AI).

### RESPONSE GUIDELINES
1. **Tone:** Professional, Confident, Humble, Energetic.
2. **Length:** SHORT and PUNCHY. Max 2-3 sentences per response. No long paragraphs.
3. **Style:** Speak in first person ("I", "Me").
4. **Consistency:** If asked about the topics in the DATA BANK, use the provided facts EXACTLY.
"""

# ==============================================================================
# ⚙️ CONFIGURATION & SETUP
# ==============================================================================

class Config:
    # 🚨 Ensure 'HF_TOKEN' is in your Streamlit Secrets!
    try:
        HF_TOKEN = st.secrets["HF_TOKEN"]
    except:
        st.error("🚨 HF_TOKEN not found in secrets. Please add it to deploy.")
        st.stop()
    
    # Using 'base' model for speed. 'Large' is too slow for real-time conversation.
    MODEL_STT = "openai/whisper-base.en" 
    APP_TITLE = "Mahesh | AI Candidate Agent"
    APP_ICON = "🤖"

# ==============================================================================
# 🔊 AUDIO ENGINE (OPTIMIZED FOR STABILITY)
# ==============================================================================

class AudioEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)

    def listen(self, audio_path):
        """Transcribe audio using HuggingFace Whisper (Fast)."""
        try:
            start_t = time.time()
            # automatic_speech_recognition is the standard HF pipeline
            response = self.client.automatic_speech_recognition(
                audio_path, 
                model=Config.MODEL_STT
            )
            return response.text, (time.time() - start_t)
        except Exception as e:
            return f"[ERROR] {str(e)}", 0.0

    def speak(self, text):
        """
        Convert text to speech using gTTS.
        Why gTTS? It is synchronous and thread-safe. 
        It effectively eliminates Streamlit Cloud asyncio crashes.
        """
        try:
            start_t = time.time()
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name

            # Generate Audio
            tts = gTTS(text=text, lang='en', tld='com', slow=False)
            tts.save(tmp_path)
            
            # Read back bytes
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            
            # Cleanup
            os.unlink(tmp_path)
            
            return audio_bytes, (time.time() - start_t)
            
        except Exception as e:
            st.error(f"Audio Generation Error: {e}")
            return None, 0.0

# ==============================================================================
# 🧠 BRAIN ENGINE (LLM LOGIC)
# ==============================================================================

class BrainEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.3" # Reliable, fast, smart

    def think(self, question):
        """Generate response as Mahesh."""
        try:
            start_t = time.time()
            
            messages = [
                {"role": "system", "content": MAHESH_PERSONA},
                {"role": "user", "content": question}
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_id,
                max_tokens=150, # Limit tokens to force brevity
                temperature=0.6 # Balance between creativity and strictness
            )
            
            answer = response.choices[0].message.content.strip()
            return answer, (time.time() - start_t)
            
        except Exception as e:
            return "I am currently experiencing high traffic. Could you ask that again?", 0.0

# ==============================================================================
# 🖥️ STREAMLIT UI (MODERN CHAT STYLE)
# ==============================================================================

def main():
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="centered"
    )

    # Initialize Session State
    if "brain" not in st.session_state:
        st.session_state.brain = BrainEngine()
        st.session_state.audio = AudioEngine()
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm Mahesh's AI Agent. You can ask me about my engineering background, my coding skills, or my life story. How can I help?"}
        ]

    # --- HEADER ---
    st.title("🤖 Mahesh AI Agent")
    st.caption("Powered by Mistral-7B & Whisper | Voice & Text Enabled")

    # --- SIDEBAR INFO ---
    with st.sidebar:
        st.image("https://api.dicebear.com/9.x/avataaars/svg?seed=Mahesh&clothing=blazerAndShirt", width=150)
        st.markdown("### 👨‍💻 Candidate Profile")
        st.info(
            """
            **Name:** Mahesh
            **Role:** AI Engineer
            **Specialty:** Agentic AI & Deployment
            **Status:** Ready to Join
            """
        )
        if st.button("🔄 Reset Conversation"):
            st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm Mahesh's AI Agent. Ready for your questions."}]
            st.rerun()

    # --- CHAT HISTORY DISPLAY ---
    # This renders the chat history using native Streamlit chat bubbles
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # --- INPUT HANDLING ---
    
    # 1. Voice Input (Top)
    audio_val = st.audio_input("🎤 Record a question (or type below)")
    
    # 2. Text Input (Bottom)
    text_val = st.chat_input("💬 Type your question here...")

    # Logic to determine which input to use
    user_input = None
    input_type = None

    if audio_val:
        with st.spinner("🎧 Transcribing audio..."):
            # Write temp file for Whisper
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_val.getvalue())
                tmp_path = tmp.name
            
            transcription, t_time = st.session_state.audio.listen(tmp_path)
            os.unlink(tmp_path)
            
            # Only process if we haven't processed this specific audio buffer yet
            # (Simple check to prevent re-running on redraw)
            if transcription and "[ERROR]" not in transcription:
                user_input = transcription
                input_type = "voice"

    if text_val:
        user_input = text_val
        input_type = "text"

    # --- PROCESSING LOOP ---
    if user_input:
        # 1. Display User Message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # 2. Generate AI Response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                answer, t_time = st.session_state.brain.think(user_input)
                
                # Show text response
                st.write(answer)
                
                # Generate Audio response
                audio_bytes, s_time = st.session_state.audio.speak(answer)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                
                # Append to history
                st.session_state.messages.append({"role": "assistant", "content": answer})

        # Force a rerun to clear inputs if needed, though chat_input clears auto
        if input_type == "voice":
            # Optional: Add a 'processed' state to avoid loop
            pass

if __name__ == "__main__":
    main()
