---
title: GAIA agent Framework 
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

# GAIA Benchmark Agent

## How to Run

### 1. Set Environment Variables
```bash
export HF_TOKEN="your_huggingface_token"
export TAVILY_API_KEY="your_tavily_api_key"
....more
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Agent

**Option A: Web Interface (Gradio)**
```bash
python app.py
```
Then open the displayed URL and:
1. Login with your Hugging Face account
2. Click "Run Evaluation & Submit All Answers"


