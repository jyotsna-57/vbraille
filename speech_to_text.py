"""
==============================================================
  ViBroBraille - Speech to Text Module (Phase 1)
==============================================================

INSTALL:
    pip install openai-whisper torch torchaudio sounddevice numpy scipy

WHISPER MODEL SIZES:
    tiny   → fastest, least accurate  (~75MB)
    base   → good balance             (~150MB)
    small  → better accuracy          (~500MB)  ← recommended
    medium → near human-level         (~1.5GB)
    large  → best accuracy            (~3GB)
==============================================================
"""

import os
import time
import wave
import tempfile
import traceback
import threading
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000
CHANNELS          = 1
CHUNK_DURATION    = 0.1
SILENCE_THRESHOLD = 0.01
SILENCE_TIMEOUT   = 2.0


class SpeechToText:

    def __init__(self, model_size: str = "small", device: str = "auto"):
        self.model_size = model_size
        self.device     = self._resolve_device(device)
        self.model      = None
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self):
        try:
            import whisper
            print(f"[STT] Loading Whisper '{self.model_size}' on {self.device}...")
            print(f"[STT] First run downloads model — please wait.")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"[STT] ✓ Whisper ready.")
        except ImportError:
            print("[STT] ⚠ Whisper not installed. Using mock transcriber.")
            print("[STT]   Install: pip install openai-whisper torch")
            self.model = None
        except Exception as e:
            print(f"[STT] ✗ Failed to load model: {e}")
            self.model = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str, language: str = "en") -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        start_time = time.time()
        if self.model is not None:
            return self._transcribe_whisper(audio_path, language, start_time)
        else:
            return self._mock_transcribe(audio_path, start_time)

    def transcribe_bytes(self, audio_bytes: bytes, fmt: str = "wav") -> dict:
        suffix   = f".{fmt}"
        tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp_file.name
        try:
            tmp_file.write(audio_bytes)
            tmp_file.close()
            return self.transcribe(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def record_and_transcribe(
        self,
        max_duration:      float = 30.0,
        auto_stop_silence: bool  = True,
        silence_timeout:   float = SILENCE_TIMEOUT
    ) -> dict:
        audio_data = self._record_audio(
            max_duration=max_duration,
            auto_stop_silence=auto_stop_silence,
            silence_timeout=silence_timeout
        )
        if audio_data is None or len(audio_data) == 0:
            return {
                "text": "", "confidence": 0.0, "language": "en",
                "duration": 0, "latency_ms": 0, "segments": [],
                "words": [], "mock": False, "error": "No audio captured"
            }
        tmp_path = self._save_wav(audio_data, SAMPLE_RATE)
        try:
            result = self.transcribe(tmp_path)
            result["recording_duration"] = len(audio_data) / SAMPLE_RATE
            return result
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ── Whisper Core ───────────────────────────────────────────────────────────

    def _transcribe_whisper(self, audio_path: str, language: str, start_time: float) -> dict:
        try:
            decode_options = dict(
                fp16=                      (self.device == "cuda"),
                verbose=                   False,
                temperature=               (0.0, 0.2, 0.4),
                word_timestamps=           True,
                condition_on_previous_text=True,
                language=                  language or "en",
                logprob_threshold=         -0.8,
                no_speech_threshold=       0.4,
                initial_prompt=            "Clear spoken English.",
            )

            result     = self.model.transcribe(audio_path, **decode_options)
            text       = result.get("text", "").strip()
            confidence = self._compute_confidence(result)
            words      = self._extract_words(result)
            latency_ms = round((time.time() - start_time) * 1000, 2)

            return {
                "text":       text,
                "confidence": confidence,
                "language":   result.get("language", language or "en"),
                "duration":   result.get("duration", 0.0),
                "latency_ms": latency_ms,
                "segments":   result.get("segments", []),
                "words":      words,
                "mock":       False
            }

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"Whisper transcription failed: {e}")

    def _compute_confidence(self, whisper_result: dict) -> float:
        segments = whisper_result.get("segments", [])
        if not segments:
            return 0.5
        scores = []
        for seg in segments:
            avg_logprob    = seg.get("avg_logprob",    -1.0)
            no_speech_prob = seg.get("no_speech_prob",  0.0)
            raw_score      = min(1.0, max(0.0, avg_logprob + 1.0))
            penalized      = raw_score * (1.0 - no_speech_prob * 0.5)
            scores.append(penalized)
        return round(sum(scores) / len(scores), 4)

    def _extract_words(self, whisper_result: dict) -> list:
        words = []
        for seg in whisper_result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word":        w.get("word", "").strip(),
                    "start":       round(w.get("start", 0.0), 3),
                    "end":         round(w.get("end",   0.0), 3),
                    "probability": round(w.get("probability", 0.0), 4)
                })
        return words

    # ── Microphone Recording ───────────────────────────────────────────────────

    def _record_audio(self, max_duration: float, auto_stop_silence: bool, silence_timeout: float) -> np.ndarray:
        try:
            import sounddevice as sd
        except ImportError:
            print("[STT] sounddevice not installed. Install: pip install sounddevice")
            return None

        chunk_samples  = int(SAMPLE_RATE * CHUNK_DURATION)
        max_chunks     = int(max_duration / CHUNK_DURATION)
        silence_chunks = int(silence_timeout / CHUNK_DURATION)
        all_audio      = []
        silent_count   = 0
        stop_flag      = threading.Event()

        def wait_for_enter():
            input()
            stop_flag.set()

        threading.Thread(target=wait_for_enter, daemon=True).start()
        print(f"[STT] 🎙 Recording... (max {max_duration}s, press Enter to stop early)")

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
                for _ in range(max_chunks):
                    if stop_flag.is_set():
                        break
                    chunk, _ = stream.read(chunk_samples)
                    chunk    = chunk.flatten()
                    all_audio.append(chunk)
                    if auto_stop_silence:
                        rms = float(np.sqrt(np.mean(chunk ** 2)))
                        if rms < SILENCE_THRESHOLD:
                            silent_count += 1
                            if silent_count >= silence_chunks:
                                print("[STT] Silence detected — stopping.")
                                break
                        else:
                            silent_count = 0
        except Exception as e:
            print(f"[STT] Recording error: {e}")
            return None

        stop_flag.set()
        if not all_audio:
            return None

        audio = np.concatenate(all_audio, axis=0)
        print(f"[STT] Recorded {len(audio)/SAMPLE_RATE:.1f}s of audio.")
        return audio

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _save_wav(self, audio: np.ndarray, sample_rate: int) -> str:
        tmp      = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        return tmp_path

    def _mock_transcribe(self, audio_path: str, start_time: float) -> dict:
        import random
        latency_ms = round((time.time() - start_time) * 1000, 2)
        mock_sentences = [
            "Hello, this is a test transcription from the mock engine.",
            "The quick brown fox jumps over the lazy dog.",
            "Speech to Braille conversion is working correctly.",
            "Welcome to ViBroBraille, your assistive communication tool.",
        ]
        return {
            "text":       random.choice(mock_sentences),
            "confidence": round(random.uniform(0.78, 0.96), 4),
            "language":   "en",
            "duration":   5.0,
            "latency_ms": latency_ms,
            "segments":   [],
            "words":      [],
            "mock":       True
        }

    def is_ready(self) -> bool:
        return self.model is not None

    def get_info(self) -> dict:
        return {
            "model_size": self.model_size,
            "device":     self.device,
            "ready":      self.is_ready(),
            "mock_mode":  not self.is_ready()
        }


# ── Standalone Test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stt = SpeechToText(model_size="small")
    print(f"Engine info: {stt.get_info()}\n")

    choice = input("1 → Mic  |  2 → File  |  3 → Mock\nChoice: ").strip()

    if choice == "1":
        result = stt.record_and_transcribe(max_duration=15.0)
    elif choice == "2":
        path   = input("File path: ").strip()
        result = stt.transcribe(path)
    else:
        result = stt._mock_transcribe("fake.wav", time.time())

    print(f"\n  Text:       {result['text']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Language:   {result['language']}")
    print(f"  Latency:    {result['latency_ms']} ms")