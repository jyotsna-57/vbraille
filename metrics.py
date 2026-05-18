"""
==============================================================
  ViBroBraille - Metrics Tracker
==============================================================

WHAT THIS MEASURES:
  - Word Error Rate (WER): how different is the corrected text from original
  - Processing latency: how long each stage takes
  - Confidence statistics: average confidence over session

INSTALL:
  pip install jiwer
"""

import time


class MetricsTracker:
    """
    Tracks Word Error Rate (WER), latency, and session statistics.

    WER Explanation:
      WER = (Substitutions + Insertions + Deletions) / Total Words in Reference
      0.0 = perfect, 1.0 = completely wrong
    """

    def __init__(self):
        self._session_wer_total   = 0.0
        self._session_count       = 0
        self._session_latency_sum = 0.0
        self._jiwer_available     = False
        self._load_jiwer()

    def _load_jiwer(self):
        try:
            import jiwer
            self._jiwer = jiwer
            self._jiwer_available = True
            print("[Metrics] jiwer loaded successfully.")
        except ImportError:
            print("[Metrics] jiwer not installed → WER will be estimated.")
            print("[Metrics] Install with: pip install jiwer")

    def evaluate(self, original: str, corrected: str) -> dict:
        """
        Compute WER between original STT output and NLP-corrected output.

        Args:
            original:  Raw text from speech-to-text
            corrected: Text after NLP correction

        Returns:
            dict with wer, latency_ms, session_avg_wer
        """
        start = time.time()

        if self._jiwer_available:
            wer = self._compute_wer_jiwer(original, corrected)
        else:
            wer = self._estimate_wer(original, corrected)

        latency_ms = round((time.time() - start) * 1000, 2)

        # Update session stats
        self._session_wer_total   += wer
        self._session_latency_sum += latency_ms
        self._session_count       += 1

        return {
            "wer":             round(wer, 4),
            "latency_ms":      latency_ms,
            "session_avg_wer": round(
                self._session_wer_total / max(1, self._session_count), 4
            ),
            "session_count":   self._session_count
        }

    def _compute_wer_jiwer(self, reference: str, hypothesis: str) -> float:
        """Use jiwer library to compute real WER."""
        try:
            return self._jiwer.wer(reference, hypothesis)
        except Exception:
            return self._estimate_wer(reference, hypothesis)

    def _estimate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Simple WER estimate without jiwer.
        Computes word-level edit distance / reference word count.
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if not ref_words:
            return 0.0

        # Count mismatched words (simplified)
        mismatches = sum(
            1 for r, h in zip(ref_words, hyp_words) if r != h
        )
        length_diff = abs(len(ref_words) - len(hyp_words))
        wer = (mismatches + length_diff) / len(ref_words)
        return min(1.0, wer)

    def get_session_stats(self) -> dict:
        """Return aggregate stats for the current session."""
        return {
            "total_utterances":  self._session_count,
            "avg_wer":           round(
                self._session_wer_total / max(1, self._session_count), 4
            ),
            "avg_latency_ms":    round(
                self._session_latency_sum / max(1, self._session_count), 2
            )
        }
