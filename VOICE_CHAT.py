"""
================================================================================
PROJECT: MAHESH AI VOICE AGENT - STAGE 1 SUBMISSION
VERSION: 10.0 (REFINED FOR WAITLIST PROGRESSION)
AUTHOR: Mahesh
DESCRIPTION:
    Voice bot that answers personality questions AS MAHESH using:
    - HuggingFace Chat API (Free ChatGPT alternative)
    - Whisper for Speech-to-Text
    - Edge-TTS/gTTS for Text-to-Speech
    
    IMPROVEMENTS (v10.0):
    1. STRICTER PERSONA: Added Hard Stop Rule for non-relevant questions.
    2. UX: Added automated welcome message on first load.
    3. SPEED: Optimized model selection for faster initial connection.
    4. ROBUSTNESS: Enhanced error logging and cleanup.
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

# 🔊 CRITICAL FIX: Apply nest_asyncio at module level (before any async calls)
# This is necessary for running asyncio within Streamlit's environment.
nest_asyncio.apply()

# ==============================================================================
# MAHESH'S PERSONA DATABASE (THE 5 KEY ANSWERS)
# ==============================================================================

MAHESH_PERSONA = """You are MAHESH - a highly professional and enthusiastic Generative AI Engineer candidate answering interview questions about yourself.

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
- Answer AS MAHESH (first person: "I", "my", "me").
- Be confident, authentic, and professional.
- **Conciseness is Critical:** Keep responses to 2-4 sentences max.
- Reference specific details from your story above.
- **HARD STOP RULE:** If the user asks a non-interview/non-personality question (e.g., "Tell me a joke," "What is 2+2?"), politely decline by saying: **"I'm here to discuss my professional background and goals, not general knowledge. Please ask me one of the core interview questions."**
"""

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # 🚨 SECURITY: Always use Streamlit secrets - NEVER hardcode tokens
    # Assume st.secrets["HF_TOKEN"] is configured.
    
    MODEL_STT = "openai/whisper-large-v3-turbo"
    VOICE_MALE = "en-US-ChristopherNeural"
    
    APP_TITLE = "Mahesh AI Voice Agent - Stage 1"
    APP_ICON = "🎙️"

# ==============================================================================
# AUDIO ENGINE (FIXED FOR STREAMLIT CLOUD)
# ==============================================================================

class AudioEngine:
    def __init__(self, token):
        self.client = InferenceClient(token=token)

    def listen(self, audio_path):
        """Transcribe audio using Whisper."""
        try:
            start_t = time.time()
            # Use a smaller/faster Whisper model as primary for quick transcription
            response = self.client.automatic_speech_recognition(
                audio_path, model="openai/whisper-small" # Optimized for speed
            )
            return response.text, (time.time() - start_t)
        except Exception as e:
            st.error(f"STT Error: {str(e)}")
            return f"[ERROR] Transcription failed: {str(e)[:50]}", 0.0

    async def _generate_speech(self, text, output_file):
        """Generate speech using Edge TTS (High Quality)."""
        communicate = edge_tts.Communicate(text, Config.VOICE_MALE)
        await communicate.save(output_file)
    
    def _generate_speech_gtts(self, text, output_file):
        """Fallback: Generate speech using Google TTS (Reliable Cloud Backup)."""
        tts = gTTS(text=text, lang='en', slow=False, tld='com')
        tts.save(output_file)

    def speak(self, text):
        """Convert text to speech - DUAL ENGINE (Edge-TTS + gTTS fallback)."""
        if "[ERROR]" in text or not text.strip(): 
            return None, 0.0
        
        start_t = time.time()
        tmp_path = None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name

            edge_tts_success = False
            
            # --- TRY METHOD 1: Edge-TTS (High Quality) ---
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            try:
                # Use a thread executor for sync running of async task (Streamlit compatible)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self._generate_speech(text, tmp_path)
                    )
                    future.result(timeout=10)
                
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
                    edge_tts_success = True
                
            except Exception as e:
                # Fall through to gTTS if Edge-TTS fails
                if os.path.exists(tmp_path): os.unlink(tmp_path)
                st.info(f"Edge-TTS failed. Using gTTS fallback.")
            
            # --- METHOD 2: gTTS Fallback ---
            if not edge_tts_success:
                self._generate_speech_gtts(text, tmp_path)
            
            # Read and cleanup
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            
            if os.path.exists(tmp_path): os.unlink(tmp_path)
            
            if len(audio_bytes) < 100:
                raise Exception("Generated audio file is too small")
            
            return audio_bytes, (time.time() - start_t)
            
        except Exception as e:
            st.error(f"🔊 CRITICAL VOICE FAILURE: {str(e)}")
            if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
            return None, 0.0

# ==============================================================================
# BRAIN ENGINE (HUGGINGFACE CHAT API)
# ==============================================================================

class BrainEngine:
    def __init__(self, token):
        self.client = InferenceClient(token=token)
        # Prioritize fast models first
        self.model_candidates = [
            "microsoft/Phi-3.5-mini-instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta"
        ]
        self.model_id = None
        self.test_connection()
        
    def test_connection(self):
        """Test and set the fastest working chat model."""
        for model in self.model_candidates:
            try:
                self.client.chat_completion(
                    messages=[{"role": "user", "content": "Hello"}],
                    model=model,
                    max_tokens=10
                )
                self.model_id = model
                return True
            except:
                continue
        self.model_id = None
        return False

    def think(self, question):
        """Generate response as Mahesh."""
        if not self.model_id:
            # Attempt to re-establish connection if offline
            if not self.test_connection():
                return "I'm having connection issues. The AI brain is currently offline.", 0.0
        
        try:
            start_t = time.time()
            
            messages = [
                {"role": "system", "content": MAHESH_PERSONA},
                {"role": "user", "content": question}
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_id,
                max_tokens=200,
                temperature=0.4 # Lower temperature for stable, predictable answers
            )
            
            answer = response.choices[0].message.content.strip()
            
            return answer, (time.time() - start_t)
            
        except Exception as e:
            st.error(f"LLM Processing Error: {str(e)}")
            return "I encountered a processing error while thinking. Please try again.", 0.0

# ==============================================================================
# STREAMLIT UI
# ==============================================================================

def main():
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="centered"
    )
    
    # --- Custom Styling (Ensuring Mobile Compatibility & Clean Look) ---
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* Unified Blue/Purple Gradient Box */
        .gradient-box {
            background: linear-gradient(135deg, #4d57a3 0%, #5b5585 100%);
            color: white;
            padding: 18px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .gradient-box b {
            color: #ffcc00; /* Yellow highlight */
        }

        /* Conversation Boxes using unified style with distinct borders */
        .question-box {
            background: linear-gradient(135deg, #4d57a3 0%, #5b5585 100%); 
            color: white;
            padding: 15px;
            border-left: 4px solid #ffcc00; /* User Question: Yellow */
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .answer-box {
            background: linear-gradient(135deg, #4d57a3 0%, #5b5585 100%); 
            color: white;
            padding: 15px;
            border-left: 4px solid #32cd32; /* Agent Answer: Green */
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Get Token from secrets
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except KeyError:
        st.error("🚨 Configuration Error: HF_TOKEN not found in Streamlit secrets.")
        st.stop()

    # Initialize Engines
    if "brain" not in st.session_state:
        with st.spinner("🚀 Initializing Mahesh's Brain and Audio Engines..."):
            st.session_state.brain = BrainEngine(hf_token)
            st.session_state.audio = AudioEngine(hf_token)
            st.session_state.history = []
            
            if st.session_state.brain.model_id:
                st.success(f"✅ System Ready! Using {st.session_state.brain.model_id.split('/')[-1]}.")
            else:
                st.error("⚠️ Connection Error: Could not connect to any LLM. Please check logs.")
            
            time.sleep(1.0)
            st.rerun()

    # --- Header ---
    st.markdown("""
        <div class='main-header'>
            <h1>🎙️ Mahesh AI Voice Agent</h1>
            <p>Stage 1 Interview Submission | Voice-Enabled Q&A Bot</p>
        </div>
    """, unsafe_allow_html=True)
    
    # --- Auto Welcome Message (UX Improvement) ---
    if not st.session_state.history and st.session_state.brain.model_id:
        welcome_message = "Welcome! I am Mahesh's voice agent. I'm prepared to answer your questions about my professional background, goals, and experience. Please begin by asking one of the core interview questions."
        
        st.markdown(f"""
            <div class='answer-box'>
                <b>🎙️ MAHESH (Welcome):</b><br>{welcome_message}
            </div>
        """, unsafe_allow_html=True)
        
        # Pre-generate and autoplay welcome audio once
        st.session_state.history.append({"role": "mahesh", "content": welcome_message, "is_welcome": True})
        
        try:
            welcome_audio_bytes, _ = st.session_state.audio.speak(welcome_message)
            if welcome_audio_bytes:
                 st.audio(welcome_audio_bytes, format="audio/mp3", autoplay=True)
        except Exception:
            pass # Fail silently if welcome audio fails

    # --- Sidebar ---
    with st.sidebar:
        st.title("📋 Assignment Details")
        st.markdown(f"**Current LLM:** **`{st.session_state.brain.model_id.split('/')[-1] if st.session_state.brain.model_id else 'Offline'}`**")
        st.markdown("---")
        
        if st.button("🗑️ Clear Conversation History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 👤 Mahesh's Core Persona")
        st.markdown("*   Mechanical Eng. → AI Dev.")
        st.markdown("*   Superpower: **Systematic Problem-Solving**")
        st.markdown("*   Growth Focus: **Agentic AI & Scale**")
        st.markdown("*   Boundary Pushing: **Weekly Prototypes**")
        
    # --- Example Questions Section ---
    with st.expander("📝 Example Core Questions", expanded=False): # Keep collapsed for clean UX
        st.markdown("""
        <div class='gradient-box'>
        <b>Ask these to test the persona:</b>
        
        1. "What should I know about your life story in a few sentences?"
        2. "What's your number one superpower?"
        3. "What are the top 3 areas you'd like to grow in?"
        4. "What misconception do your coworkers have about you?"
        5. "How do you push your boundaries and limits?"
        </div>
        """, unsafe_allow_html=True)

    # --- Conversation History ---
    if st.session_state.history:
        st.markdown("### 💬 Conversation")
        
        for msg in st.session_state.history:
            # Skip re-rendering the welcome message text, as it's static above
            if msg.get("is_welcome"):
                continue 
                
            if msg['role'] == 'user':
                st.markdown(f"""
                    <div class='question-box'>
                        <b>🧑 YOU:</b><br>{msg['content']}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='answer-box'>
                        <b>🎙️ MAHESH:</b><br>{msg['content']}
                    </div>
                """, unsafe_allow_html=True)
        
        st.divider()

    # --- Voice Input & Processing ---
    st.markdown("### 🎤 Start Speaking")
    audio_input = st.audio_recorder("Click to Record Question", icon="🎙️", sample_rate=16000)

    if audio_input:
        with st.status("⚡ Processing Question...", expanded=True) as status:
            
            # 1. Speech to Text
            st.write("👂 Listening and Transcribing...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_input)
                tmp_path = tmp.name

            question, listen_time = st.session_state.audio.listen(tmp_path)
            os.unlink(tmp_path)
            
            if "[ERROR]" in question:
                st.error("Transcription failed. Please record again.")
                status.update(label="❌ Failed", state="error")
                st.stop()
            
            # Record user question immediately
            st.session_state.history.append({"role": "user", "content": question})

            st.write(f"✅ Question Transcribed: **{question}**")

            # 2. Generate Answer
            st.write("🧠 Thinking (Generating Mahesh's Persona Answer)...")
            answer, think_time = st.session_state.brain.think(question)
            
            if not answer or len(answer) < 5:
                st.error("Failed to generate response.")
                status.update(label="❌ Failed", state="error")
                st.stop()

            # Record agent answer immediately
            st.session_state.history.append({"role": "mahesh", "content": answer})

            st.write(f"✅ Answer Generated!")
            
            # 3. Text to Speech
            st.write("🗣️ Generating Voice Output...")
            audio_bytes, speak_time = st.session_state.audio.speak(answer)
            
            if audio_bytes:
                status.update(label="✅ Complete!", state="complete")
            else:
                status.update(label="⚠️ Voice generation failed (Text displayed)", state="error")
                
        # --- Rerun to Display History and Play Audio ---
        st.rerun()

    # --- Display Audio (Run after history update and status block) ---
    if st.session_state.history and st.session_state.history[-1]['role'] == 'mahesh':
        # Find the last response's text to regenerate audio if needed (e.g. on rerun)
        last_answer = st.session_state.history[-1]['content']
        
        # Regenerate audio in main block if not already generated, or just play
        if 'last_audio' not in st.session_state or st.session_state.history[-1] != st.session_state.get('last_message'):
            # Only generate/play if the last message was a new agent response
            try:
                # The heavy lifting is done in the audio_input block, this is just for display/playback
                audio_bytes, speak_time = st.session_state.audio.speak(last_answer)
                st.session_state['last_audio'] = audio_bytes
                st.session_state['last_message'] = st.session_state.history[-1]
            except Exception:
                st.session_state['last_audio'] = None
        
        # Display/Play
        if st.session_state.get('last_audio'):
            st.audio(st.session_state['last_audio'], format="audio/mp3", autoplay=True)
            
            # Display metrics if we just ran through the pipeline (approximate metrics)
            if 'listen_time' in locals() and 'think_time' in locals() and 'speak_time' in locals():
                 with st.expander("⚡ Performance Metrics"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Listen", f"{listen_time:.1f}s")
                    col2.metric("Think", f"{think_time:.1f}s")
                    col3.metric("Speak", f"{speak_time:.1f}s")
                    col4.metric("Total", f"{listen_time+think_time+speak_time:.1f}s")

# --- Initial Run ---
if __name__ == "__main__":
    main()
