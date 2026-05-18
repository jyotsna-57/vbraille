"""
==============================================================
  ViBroBraille - Data Logger
==============================================================

Logs all pipeline outputs to CSV files for research analysis.

Log files created in /logs/ directory:
  transcriptions.csv → STT outputs + confidence
  braille_outputs.csv → corrected text + Braille + metrics
  pipeline_full.csv   → full end-to-end pipeline logs

NO EXTRA DEPENDENCIES: Uses Python stdlib csv + datetime.
"""

import csv
import os
import json
from datetime import datetime


LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


class AppLogger:
    """Logs pipeline data to CSV files for research analysis."""

    def __init__(self):
        os.makedirs(LOGS_DIR, exist_ok=True)
        self._init_csv_files()

    def _init_csv_files(self):
        """Create CSV files with headers if they don't exist."""
        files = {
            "transcriptions.csv": [
                "timestamp", "text", "confidence", "language",
                "duration_s", "latency_ms", "mock"
            ],
            "braille_outputs.csv": [
                "timestamp", "original_text", "corrected_text",
                "braille_unicode", "cell_count", "confidence",
                "low_confidence", "steps", "latency_ms"
            ],
            "pipeline_full.csv": [
                "timestamp", "raw_text", "corrected_text",
                "braille_unicode", "confidence", "wer",
                "total_latency_ms"
            ]
        }
        for filename, headers in files.items():
            path = os.path.join(LOGS_DIR, filename)
            if not os.path.exists(path):
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)

    def log_transcription(self, stt_result: dict):
        """Log Phase 1 STT result."""
        path = os.path.join(LOGS_DIR, "transcriptions.csv")
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                stt_result.get("text", ""),
                stt_result.get("confidence", ""),
                stt_result.get("language", "en"),
                stt_result.get("duration", ""),
                stt_result.get("latency_ms", ""),
                stt_result.get("mock", False)
            ])

    def log_braille_output(self, nlp_result: dict, braille_result: dict):
        """Log Phase 2 NLP + Braille result."""
        path = os.path.join(LOGS_DIR, "braille_outputs.csv")
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                nlp_result.get("original_text", ""),
                nlp_result.get("corrected_text", ""),
                braille_result.get("unicode", ""),
                braille_result.get("cell_count", ""),
                nlp_result.get("confidence", ""),
                nlp_result.get("low_confidence", ""),
                json.dumps(nlp_result.get("steps", [])),
                nlp_result.get("latency_ms", "")
            ])

    def log_full_pipeline(self, data: dict):
        """Log complete pipeline run."""
        path = os.path.join(LOGS_DIR, "pipeline_full.csv")
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                data.get("raw_text", ""),
                data.get("nlp", {}).get("corrected_text", ""),
                data.get("braille", {}).get("unicode", ""),
                data.get("nlp", {}).get("confidence", ""),
                data.get("wer", ""),
                data.get("latency_ms", "")
            ])
