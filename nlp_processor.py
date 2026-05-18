"""
==============================================================
  ViBroBraille - Phase 2: NLP Processor
==============================================================

WHAT THIS MODULE DOES:
  After speech-to-text (Phase 1), raw transcriptions often have:
    - Grammar errors          → "i went to store"
    - Punctuation missing     → "hello world how are you"
    - Misheard words          → "their" vs "there"
    - Filler words            → "um", "uh", "like"

  This module fixes all of the above before Braille encoding.

PIPELINE INSIDE THIS MODULE:
  Raw STT Text
      ↓
  1. Filler word removal     (regex)
      ↓
  2. Basic grammar correction (rule-based)
      ↓
  3. Transformer correction   (HappyTransformer / T5 grammar model)
      ↓
  4. Confidence warning       (flag low-confidence transcriptions)
      ↓
  Corrected Text → Braille Encoder

INSTALL DEPENDENCIES:
  pip install happytransformer spacy
  python -m spacy download en_core_web_sm

  NOTE: happytransformer downloads a grammar correction model (~250MB)
        on first use. This uses T5-based grammar correction.

FALLBACK:
  If transformers aren't installed, rule-based correction only is used.
"""

import re
import time
import traceback


# Confidence threshold below which we warn the user
CONFIDENCE_WARNING_THRESHOLD = 0.70


class NLPProcessor:
    """
    Phase 2: Cleans and corrects raw speech-to-text output before Braille encoding.

    Usage:
        nlp = NLPProcessor()
        result = nlp.process("i went to the store yestrday", confidence=0.82)
        print(result["corrected_text"])  # "I went to the store yesterday."
        print(result["low_confidence"])  # False
    """

    def __init__(self):
        self.grammar_model = None
        self._load_models()

    def _load_models(self):
        """Load NLP/grammar correction models. Gracefully falls back if missing."""
        # Try HappyTransformer grammar correction (T5)
        try:
            from happytransformer import HappyTextToText, TTSettings
            print("[NLP] Loading grammar correction model (T5)...")
            self.grammar_model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
            self.tt_settings   = TTSettings(num_beams=5, min_length=1)
            print("[NLP] Grammar model loaded.")
        except ImportError:
            print("[NLP] happytransformer not installed → using rule-based correction only.")
            print("[NLP] Install with: pip install happytransformer")
            self.grammar_model = None
        except Exception as e:
            print(f"[NLP] Could not load grammar model: {e}")
            self.grammar_model = None

    # ── Public API ────────────────────────────────────────

    def process(self, raw_text: str, confidence: float = 1.0) -> dict:
        """
        Full NLP processing pipeline.

        Args:
            raw_text:   Raw string from speech-to-text
            confidence: Confidence score from STT (0.0–1.0)

        Returns:
            dict with keys:
                corrected_text → cleaned, grammatically correct string
                low_confidence → True if confidence is below warning threshold
                steps          → list of transformation steps applied
                latency_ms     → processing time
        """
        start_time = time.time()
        steps = []

        text = raw_text.strip()

        # ── Step 1: Remove filler words ───────────────────
        text = self._remove_fillers(text)
        steps.append("filler_removal")

        # ── Step 2: Basic rule-based fixes ────────────────
        text = self._rule_based_corrections(text)
        steps.append("rule_based_correction")

        # ── Step 3: Transformer grammar correction ────────
        if self.grammar_model and len(text) > 3:
            try:
                text = self._transformer_correction(text)
                steps.append("transformer_grammar_correction")
            except Exception as e:
                print(f"[NLP] Transformer correction failed: {e}")
                steps.append("transformer_correction_skipped")

        # ── Step 4: Capitalize and finalize ───────────────
        text = self._finalize(text)
        steps.append("finalization")

        latency_ms    = round((time.time() - start_time) * 1000, 2)
        low_confidence = confidence < CONFIDENCE_WARNING_THRESHOLD

        return {
            "corrected_text": text,
            "original_text":  raw_text,
            "low_confidence": low_confidence,
            "confidence":     confidence,
            "steps":          steps,
            "latency_ms":     latency_ms
        }

    # ── Internal Steps ────────────────────────────────────

    def _remove_fillers(self, text: str) -> str:
        """
        Remove spoken filler words that add no semantic value.
        e.g. "um like uh hello there" → "hello there"
        """
        fillers = [
            r"\bum\b", r"\buh\b", r"\ber\b", r"\bhmm\b",
            r"\blike\b", r"\byou know\b", r"\bkind of\b",
            r"\bsort of\b", r"\bactually\b", r"\bbasically\b",
            r"\bright\b", r"\bokay so\b", r"\bso\b"
        ]
        for filler in fillers:
            text = re.sub(filler, "", text, flags=re.IGNORECASE)

        # Clean up multiple spaces created by removal
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def _rule_based_corrections(self, text: str) -> str:
        """
        Apply common rule-based corrections for speech recognition errors.
        These handle the most frequent STT mistakes.
        """
        # Common STT misrecognitions
        corrections = {
            r"\bi\b":          "I",           # Capitalize standalone 'i'
            r"\bim\b":         "I'm",
            r"\bive\b":        "I've",
            r"\bwouldnt\b":    "wouldn't",
            r"\bcouldnt\b":    "couldn't",
            r"\bdont\b":       "don't",
            r"\bdoesnt\b":     "doesn't",
            r"\bisnt\b":       "isn't",
            r"\barent\b":      "aren't",
            r"\bwont\b":       "won't",
            r"\bcant\b":       "can't",
            r"\bwanna\b":      "want to",
            r"\bgonna\b":      "going to",
            r"\bgotta\b":      "got to",
            r"\bkinda\b":      "kind of",
            r"\blotta\b":      "lot of",
            r"\bthere\b(?=.*\bgoing)": "their",  # context-sensitive (simplified)
        }

        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _transformer_correction(self, text: str) -> str:
        """
        Use a T5 transformer model to correct grammar.

        The model expects the prefix "grammar: " before the text.
        It returns a grammatically corrected version.

        Example:
            Input:  "grammar: i goes to the store yesterday"
            Output: "I went to the store yesterday."
        """
        # Limit input length to 512 tokens (model limit)
        if len(text) > 400:
            # Process in chunks if text is long
            return self._chunk_correct(text)

        prefixed = f"grammar: {text}"
        result   = self.grammar_model.generate_text(prefixed, args=self.tt_settings)
        return result.text.strip()

    def _chunk_correct(self, text: str, chunk_size: int = 300) -> str:
        """
        Split long text into sentences, correct each, then rejoin.
        Used when text exceeds transformer model limits.
        """
        # Simple sentence split on '. ' and '! ' and '? '
        sentences = re.split(r'(?<=[.!?])\s+', text)
        corrected_parts = []

        for sentence in sentences:
            if len(sentence) > chunk_size:
                # If a single sentence is too long, truncate
                sentence = sentence[:chunk_size]
            try:
                prefixed  = f"grammar: {sentence}"
                result    = self.grammar_model.generate_text(prefixed, args=self.tt_settings)
                corrected_parts.append(result.text.strip())
            except Exception:
                corrected_parts.append(sentence)

        return " ".join(corrected_parts)

    def _finalize(self, text: str) -> str:
        """
        Final cleanup:
        - Capitalize first letter
        - Ensure text ends with punctuation
        - Remove duplicate spaces
        """
        if not text:
            return text

        # Remove extra whitespace
        text = re.sub(r"\s{2,}", " ", text).strip()

        # Capitalize first character
        text = text[0].upper() + text[1:]

        # Add period at end if no punctuation
        if text and text[-1] not in ".!?":
            text += "."

        return text
