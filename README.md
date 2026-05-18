# ViBroBraille – Setup Guide

## Project Structure

```
vibrobraille/
├── backend/
│   ├── app.py              ← Flask server (run this)
│   ├── speech_to_text.py   ← Phase 1: Whisper STT
│   ├── nlp_processor.py    ← Phase 2: NLP grammar correction
│   ├── braille_encoder.py  ← Phase 2: Unicode Braille mapping
│   ├── metrics.py          ← WER + latency tracking
│   └── logger.py           ← CSV logging
├── frontend/
│   └── index.html          ← Full UI (open via Flask)
├── logs/                   ← Auto-created on first run
└── requirements.txt
```

---

## Step 1 – Install Python (if needed)

Make sure Python 3.9+ is installed.

```bash
python --version
```

---

## Step 2 – Create a Virtual Environment

```bash
# In the vibrobraille/ folder:
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

---

## Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

> First run downloads Whisper model (~150MB) and T5 grammar model (~250MB).
> This only happens once.

Install spaCy language model separately:

```bash
python -m spacy download en_core_web_sm
```

---

## Step 4 – Run the Server

```bash
cd backend
python app.py
```

You should see:

```
=======================================================
   ViBroBraille Server Starting...
   Open: http://localhost:5000
=======================================================
```

---

## Step 5 – Open the App

Open your browser at:

```
http://localhost:5000
```

---

## Step 6 – Test the App

### Option A: Microphone Input (Full Pipeline)
1. Click the 🎙️ mic button
2. Allow microphone access
3. Speak clearly
4. Click ⏹️ to stop
5. Watch the Braille animation render

### Option B: Text Input (Test without mic)
1. Type any text in the bottom input field
2. Click "Process →"
3. Braille output appears immediately

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server health check |
| POST | `/api/transcribe` | Phase 1 only: audio → text |
| POST | `/api/process-text` | Phase 2 only: text → NLP → Braille |
| POST | `/api/speech-to-braille` | Full pipeline: audio → Braille |

---

## Troubleshooting

**"Cannot reach server" in browser**
→ Make sure `python app.py` is running in the backend/ folder

**Whisper model not found**
→ Run: `pip install openai-whisper torch`

**Grammar model slow on first run**
→ HappyTransformer downloads T5 model once. Subsequent runs are fast.

**Mic not working in browser**
→ Chrome/Firefox require HTTPS for mic access on non-localhost.
→ Use `http://localhost:5000` (not IP address) for local development.

---

## Logs

All pipeline outputs are saved to:
- `logs/transcriptions.csv`
- `logs/braille_outputs.csv`
- `logs/pipeline_full.csv`
