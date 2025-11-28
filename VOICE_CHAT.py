"""
================================================================================
PROJECT: MAHESH AI VOICE AGENT - STAGE 1 SUBMISSION
VERSION: 10.0 (STREAMLIT-CLOUD COMPATIBLE + MALE VOICE)
AUTHOR: Mahesh
DESCRIPTION:
    Voice bot that answers personality questions AS MAHESH using:
    - HuggingFace Chat API
    - Whisper for Speech-to-Text
    - Coqui TTS for Text-to-Speech (MALE VOICE, free, works on cloud)
    - Complete persona with 5 core interview answers
================================================================================
"""

import streamlit as st
from huggingface_hub import InferenceClient
import tempfile
import time
import os
import numpy as np
import soundfile as sf
from TTS.api import TTS

# ==============================================================================
# MAHESH'S PERSONA DATABASE (THE 5 KEY ANSWERS)
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
- For other questions, stay consistent with this persona
"""

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    HF_TOKEN = st.secrets["HF_TOKEN"]  # Secure
    MODEL_STT = "openai/whisper-large-v3-turbo"
    APP_TITLE = "Mahesh AI Voice Agent - Stage 1"
    APP_ICON = "🎙️"

# ==============================================================================
# AUDIO ENGINE (COQUI TTS)
# ==============================================================================

class AudioEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)

        # Load Coqui TTS model (MALE VOICE p243)
        self.tts = TTS("tts_models/en/vctk/vits")

    def listen(self, audio_path):
        """Transcribe audio using Whisper."""
        try:
            start = time.time()
            response = self.client.automatic_speech_recognition(
                audio_path, model=Config.MODEL_STT
            )
            return response.text, (time.time() - start)
        except Exception:
            return "[ERROR] Could not transcribe audio.", 0.0

    def speak(self, text):
        """Convert text to male voice (Coqui TTS)."""
        try:
            start = time.time()

            # Generate WAV audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                output_path = tmp.name

            wav = self.tts.tts(text=text, speaker="p243")
            wav_np = np.array(wav)

            # Save file
            sf.write(output_path, wav_np, 22050)

            # Read back
            with open(output_path, "rb") as f:
                audio_bytes = f.read()

            os.unlink(output_path)
            return audio_bytes, (time.time() - start)

        except Exception as e:
            print("TTS ERROR:", e)
            return None, 0.0

# ==============================================================================
# BRAIN ENGINE (HUGGINGFACE CHAT API)
# ==============================================================================

class BrainEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)
        self.model_id = None
        self.test_connection()

    def test_connection(self):
        models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta",
            "microsoft/Phi-3.5-mini-instruct"
        ]

        for m in models:
            try:
                res = self.client.chat_completion(
                    messages=[{"role": "user", "content": "Hi"}],
                    model=m, max_tokens=5
                )
                if res and res.choices:
                    self.model_id = m
                    return True
            except:
                continue
        return False

    def think(self, question):
        if not self.model_id:
            return "I'm having connection issues. Please try again.", 0.0

        start = time.time()

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": MAHESH_PERSONA},
                    {"role": "user", "content": question}
                ],
                model=self.model_id,
                max_tokens=200,
                temperature=0.7
            )
            answer = response.choices[0].message.content.strip()
            return answer, (time.time() - start)

        except Exception:
            return "Something went wrong generating my response.", 0.0

# ==============================================================================
# STREAMLIT UI
# ==============================================================================

def main():
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="centered"
    )

    # Initialize
    if "brain" not in st.session_state:
        with st.spinner("🚀 Initializing Mahesh's Brain..."):
            st.session_state.brain = BrainEngine()
            st.session_state.audio = AudioEngine()
            st.session_state.history = []
            time.sleep(1)
            st.rerun()

    # Header
    st.markdown("""
        <div style='text-align:center;padding:20px;background:#764ba2;color:white;border-radius:10px;'>
            <h1>🎙️ Mahesh AI Voice Agent</h1>
            <p>Stage 1 Interview Submission | Voice-Enabled Q&A Bot</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📋 Assignment Info")
        st.write("**Technologies:** Whisper STT, HuggingFace LLM, Coqui TTS")
        if st.session_state.brain.model_id:
            st.success(f"Model: {st.session_state.brain.model_id.split('/')[-1]}")
        else:
            st.error("Brain Offline")
        st.button("Clear Chat", on_click=lambda: st.session_state.update({"history": []}))

    st.markdown("### 🎤 Ask Your Question")
    audio = st.audio_input("Click to record")

    if audio:
        with st.status("⚡ Processing...", expanded=True) as status:

            # SAVE AUDIO
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio.getvalue())
                path = tmp.name

            # STT
            st.write("👂 Transcribing...")
            question, t1 = st.session_state.audio.listen(path)
            os.unlink(path)

            if "[ERROR]" in question:
                st.error(question)
                status.update(state="error")
                return

            st.write(f"**You Asked:** {question}")

            # LLM
            st.write("🧠 Thinking...")
            answer, t2 = st.session_state.brain.think(question)

            # TTS
            st.write("🗣 Generating male voice...")
            audio_bytes, t3 = st.session_state.audio.speak(answer)

            status.update(state="complete")

        # Show message
        st.success("Response Ready")
        st.write(f"### 🧑 YOU:\n{question}")
        st.write(f"### 🎙 MAHESH:\n{answer}")

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav", autoplay=True)

        # Save History
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "mahesh", "content": answer})

if __name__ == "__main__":
    main()

