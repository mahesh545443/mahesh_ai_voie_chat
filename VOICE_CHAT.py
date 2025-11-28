"""
================================================================================
PROJECT: MAHESH AI VOICE AGENT - STAGE 1 SUBMISSION
VERSION: 9.3 (STREAMLIT CLOUD DEPLOYMENT FIXED)
AUTHOR: Mahesh
DESCRIPTION:
    Voice bot that answers personality questions AS MAHESH using:
    - HuggingFace Chat API (Free ChatGPT alternative)
    - Whisper for Speech-to-Text
    - Edge-TTS for Text-to-Speech
    
    DEPLOYMENT FIXES:
    1. nest_asyncio.apply() moved to global scope (Line 17)
    2. Added proper asyncio loop handling for Streamlit Cloud
    3. Fallback to get_event_loop() for better compatibility
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
nest_asyncio.apply()

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
- For other questions, stay consistent with this persona"""

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # 🚨 SECURITY: Always use Streamlit secrets - NEVER hardcode tokens
    HF_TOKEN = st.secrets["HF_TOKEN"]
    
    MODEL_STT = "openai/whisper-large-v3-turbo"
    VOICE_MALE = "en-US-ChristopherNeural"
    
    APP_TITLE = "Mahesh AI Voice Agent - Stage 1"
    APP_ICON = "🎙️"

# ==============================================================================
# AUDIO ENGINE (FIXED FOR STREAMLIT CLOUD)
# ==============================================================================

class AudioEngine:
    def __init__(self):
        self.client = InferenceClient(token=Config.HF_TOKEN)

    def listen(self, audio_path):
        """Transcribe audio using Whisper."""
        try:
            start_t = time.time()
            response = self.client.automatic_speech_recognition(
                audio_path, model=Config.MODEL_STT
            )
            return response.text, (time.time() - start_t)
        except Exception:
            try:
                response = self.client.automatic_speech_recognition(
                    audio_path, model="openai/whisper-small"
                )
                return response.text, 0.0
            except Exception as e:
                return f"[ERROR] {str(e)}", 0.0

    async def _generate_speech(self, text, output_file):
        """Generate speech using Edge TTS."""
        try:
            communicate = edge_tts.Communicate(text, Config.VOICE_MALE)
            await communicate.save(output_file)
        except Exception as e:
            raise Exception(f"Edge TTS Error: {str(e)}")
    
    def _generate_speech_gtts(self, text, output_file):
        """Fallback: Generate speech using Google TTS."""
        try:
            tts = gTTS(text=text, lang='en', slow=False, tld='com')
            tts.save(output_file)
        except Exception as e:
            raise Exception(f"gTTS Error: {str(e)}")

    def speak(self, text):
        """Convert text to speech - DUAL ENGINE (Edge-TTS + gTTS fallback)."""
        if "[ERROR]" in text: 
            return None, 0.0
        
        start_t = time.time()
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp_path = tmp.name

            # 🔊 TRY METHOD 1: Edge-TTS (Best Quality)
            edge_tts_success = False
            try:
                # Proper asyncio loop handling for Streamlit Cloud
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Handling for Streamlit's possibly running loop
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self._generate_speech(text, tmp_path)
                        )
                        future.result(timeout=10)  # 10 second timeout
                else:
                    loop.run_until_complete(self._generate_speech(text, tmp_path))
                
                # Check if file has content
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 1000:
                    edge_tts_success = True
                
            except Exception as e:
                # Cleanup if Edge-TTS failed but created a file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                st.warning(f"Edge-TTS failed: {str(e)[:50]}... Trying backup voice engine...")
            
            # 🔊 METHOD 2: gTTS Fallback (Always Works on Cloud)
            if not edge_tts_success:
                self._generate_speech_gtts(text, tmp_path)
            
            # Read the generated audio file
            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
            
            # Cleanup
            os.unlink(tmp_path)
            
            # Verify we got audio
            if len(audio_bytes) < 100:
                raise Exception("Audio file is too small")
            
            return audio_bytes, (time.time() - start_t)
            
        except Exception as e:
            st.error(f"🔊 Both voice engines failed: {str(e)}")
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
        """Test which chat model works."""
        models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta",
            "microsoft/Phi-3.5-mini-instruct"
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
        
        self.model_id = None
        return False

    def think(self, question):
        """Generate response as Mahesh."""
        if not self.model_id:
            self.test_connection()
            if not self.model_id:
                return "I'm having connection issues. Please try again.", 0.0
        
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
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            
            return answer, (time.time() - start_t)
            
        except Exception as e:
            # Try backup models
            backups = ["mistralai/Mistral-7B-Instruct-v0.2", "HuggingFaceH4/zephyr-7b-beta"]
            for backup in backups:
                try:
                    response = self.client.chat_completion(
                        messages=[
                            {"role": "system", "content": MAHESH_PERSONA},
                            {"role": "user", "content": question}
                        ],
                        model=backup,
                        max_tokens=200
                    )
                    return response.choices[0].message.content.strip(), 0.0
                except:
                    continue
            
            return "I'm having trouble responding. Please try again.", 0.0

# ==============================================================================
# STREAMLIT UI
# ==============================================================================

def main():
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon=Config.APP_ICON,
        layout="centered"
    )
    
    # Custom Styling (UPDATED TO INCLUDE INSTRUCTION BOX)
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
        .question-box {
            background: #f0f2f6; /* Light Grey */
            padding: 15px;
            border-left: 4px solid #667eea; /* Blue */
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .answer-box {
            background: #e8f5e9; /* Light Green */
            padding: 15px;
            border-left: 4px solid #4caf50; /* Green */
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .example-questions {
            /* Darker Indigo Gradient */
            background: linear-gradient(135deg, #4d57a3 0%, #5b5585 100%); 
            color: white; 
            padding: 18px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .example-questions b {
            color: #ffcc00; /* Highlight important text in yellow */
        }
        .instruction-box {
            /* Matching the Example Questions box style */
            background: linear-gradient(135deg, #4d57a3 0%, #5b5585 100%); 
            color: white; 
            padding: 18px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .instruction-box b {
            color: #ffcc00; /* Highlight important text in yellow */
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize
    if "brain" not in st.session_state:
        with st.spinner("🚀 Initializing Mahesh's Brain..."):
            st.session_state.brain = BrainEngine()
            st.session_state.audio = AudioEngine()
            st.session_state.history = []
            
            if st.session_state.brain.model_id:
                st.success("✅ System Ready!")
            else:
                st.warning("⚠️ Connection slow. Responses may be delayed.")
            
            time.sleep(1.5)
            st.rerun()

    # Header
    st.markdown("""
        <div class='main-header'>
            <h1>🎙️ Mahesh AI Voice Agent</h1>
            <p>Stage 1 Interview Submission | Voice-Enabled Q&A Bot</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("📋 Assignment Info")
        
        st.markdown("""
        **Task:** Create a voice bot that answers personality questions
        
        **Technologies:**
        - 🗣️ Voice Input: **Whisper STT**
        - 🧠 Brain: **HuggingFace Chat API**
        - 🔊 Voice Output: **Edge-TTS**
        
        **Status:**
        """)
        
        if st.session_state.brain.model_id:
            st.success(f"✅ {st.session_state.brain.model_id.split('/')[-1]}")
        else:
            st.error("❌ Offline")
        
        st.divider()
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
        
        st.divider()
        
        st.markdown("""
        ### 👤 About Mahesh
        - Mechanical Engineer → AI Dev
        - pg in data science and machine learning
        - Builds weekly prototypes
        - Specializes in Agentic AI
        """)

    # Example Questions Section
    with st.expander("📝 Example Questions to Ask", expanded=len(st.session_state.history)==0):
        st.markdown("""
        <div class='example-questions'>
        <b>Try asking these 5 core questions:</b>
        
        1. "What should I know about your life story?"
        2. "What's your number one superpower?"
        3. "What are the top 3 areas you'd like to grow in?"
        4. "What misconception do your coworkers have about you?"
        5. "How do you push your boundaries and limits?"
        
        <b>Or ask anything else:</b>
        - "Why did you transition from mechanical to AI?"
        - "What projects are you currently working on?"
        - "What's your approach to learning new technologies?"
        </div>
        """, unsafe_allow_html=True)

    # Conversation History
    if st.session_state.history:
        st.markdown("### 💬 Conversation")
        
        for msg in st.session_state.history:
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

    # Voice Input
    st.markdown("### 🎤 Ask Your Question")
    audio_input = st.audio_input("Click to record")

    if audio_input:
        with st.status("⚡ Processing...", expanded=True) as status:
            # Step 1: Speech to Text
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_input.getvalue())
                tmp_path = tmp.name

            st.write("👂 Listening...")
            question, listen_time = st.session_state.audio.listen(tmp_path)
            os.unlink(tmp_path)
            
            if "[ERROR]" in question:
                st.error(question)
                status.update(label="❌ Failed", state="error")
                st.stop()

            st.write(f"✅ Question: **{question}**")

            # Step 2: Generate Answer
            st.write("🧠 Thinking...")
            answer, think_time = st.session_state.brain.think(question)
            
            if not answer or len(answer) < 5:
                st.error("Failed to generate response")
                status.update(label="❌ Failed", state="error")
                st.stop()

            st.write(f"✅ Answer ready!")

            # Step 3: Text to Speech
            st.write("🗣️ Generating voice...")
            audio_bytes, speak_time = st.session_state.audio.speak(answer)
            
            if audio_bytes:
                status.update(label="✅ Complete!", state="complete")
            else:
                status.update(label="⚠️ Voice generation failed", state="error")

        # Display Results
        st.markdown(f"""
            <div class='question-box'>
                <b>🧑 YOU:</b><br>{question}
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='answer-box'>
                <b>🎙️ MAHESH:</b><br>{answer}
            </div>
        """, unsafe_allow_html=True)
        
        # Play Audio
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            
            # Save to history
            st.session_state.history.append({"role": "user", "content": question})
            st.session_state.history.append({"role": "mahesh", "content": answer})
            
            # Performance Metrics
            with st.expander("⚡ Performance Metrics"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Listen", f"{listen_time:.1f}s")
                col2.metric("Think", f"{think_time:.1f}s")
                col3.metric("Speak", f"{speak_time:.1f}s")
                col4.metric("Total", f"{listen_time+think_time+speak_time:.1f}s")
        else:
            st.warning("🔊 Voice output failed, but text response is shown above.")

    # Instructions for first use (UPDATED FOR LINE-BY-LINE DISPLAY)
    if not st.session_state.history:
        st.markdown("""
        <div class='instruction-box'>
        ### 🎯 How to Use:
        
        * 1. Click the **microphone button** above
        * 2. **Ask one of the 5 core questions** (see examples above)
        * 3. **Listen to Mahesh's response** in natural voice
        
        💡 **Tip:** The bot is designed to answer personality/interview questions as Mahesh would answer them.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

