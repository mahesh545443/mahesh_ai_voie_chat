# mahesh_ai_voie_chat
# Mahesh AI Voice Agent - Stage 1

Voice-enabled Q&A bot that answers personality questions using AI.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Secrets
Create `.streamlit/secrets.toml` and add:
```toml
HF_TOKEN = "your_huggingface_token_here"
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Run
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Push code to GitHub (without secrets!)
2. Connect repo to Streamlit Cloud
3. Add `HF_TOKEN` in Settings → Secrets
4. Deploy

## Technologies
- **STT:** Whisper (HuggingFace)
- **LLM:** Llama 3.2 / Mistral (HuggingFace Chat API)
- **TTS:** Edge-TTS (Microsoft)

## Features
✅ Voice input/output
✅ 5 core personality questions answered
✅ Secure token management
✅ Performance metrics
