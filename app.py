"""
ViBroBraille - Flask App
Includes: Speech-to-Braille + Image-to-Braille
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import traceback

from speech_to_text import SpeechToText
from nlp_processor import NLPProcessor
from braille_encoder import BrailleEncoder
from metrics import MetricsTracker
from logger import AppLogger
from image_processor import ImageProcessor

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ── Initialize all modules ─────────────────────────────────────────────────────
stt     = SpeechToText(model_size="small")
nlp     = NLPProcessor()
encoder = BrailleEncoder()
metrics = MetricsTracker()
logger  = AppLogger()
imgproc = ImageProcessor()


# ── Serve Frontend ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ── Health Check ───────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "message": "ViBroBraille server is running",
        "models":  imgproc.is_ready()
    })


# ── PHASE 1: Speech → Text ─────────────────────────────────────────────────────
@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        temp_path  = "temp_audio.wav"
        audio_file.save(temp_path)

        result = stt.transcribe(temp_path, language="en")

        if os.path.exists(temp_path):
            os.remove(temp_path)

        logger.log_transcription(result)

        return jsonify({
            "recognized_text": result["text"],
            "confidence":      result["confidence"],
            "language":        result.get("language", "en"),
            "phase":           1
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── PHASE 2: Text → NLP → Braille ─────────────────────────────────────────────
@app.route("/api/process-text", methods=["POST"])
def process_text():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        raw_text   = data["text"]
        confidence = data.get("confidence", 1.0)

        nlp_result     = nlp.process(raw_text, confidence)
        braille_output = encoder.encode(nlp_result["corrected_text"])
        metric_result  = metrics.evaluate(
            original=raw_text,
            corrected=nlp_result["corrected_text"]
        )

        logger.log_braille_output(nlp_result, braille_output)

        return jsonify({
            "original_text":  raw_text,
            "corrected_text": nlp_result["corrected_text"],
            "braille":        braille_output["unicode"],
            "braille_cells":  braille_output["cells"],
            "confidence":     confidence,
            "low_confidence": nlp_result["low_confidence"],
            "wer":            metric_result["wer"],
            "latency_ms":     metric_result["latency_ms"],
            "phase":          2
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── FULL PIPELINE: Speech → Text → NLP → Braille ──────────────────────────────
@app.route("/api/speech-to-braille", methods=["POST"])
def speech_to_braille():
    try:
        import time
        start_time = time.time()

        if "audio" in request.files:
            audio_file = request.files["audio"]
            temp_path  = "temp_pipeline.wav"
            audio_file.save(temp_path)

            stt_result = stt.transcribe(temp_path, language="en")

            if os.path.exists(temp_path):
                os.remove(temp_path)

            raw_text   = stt_result["text"]
            confidence = stt_result["confidence"]

        elif request.is_json:
            data       = request.get_json()
            raw_text   = data.get("text", "")
            confidence = data.get("confidence", 1.0)
        else:
            return jsonify({"error": "Provide audio file or JSON text body"}), 400

        nlp_result     = nlp.process(raw_text, confidence)
        braille_output = encoder.encode(nlp_result["corrected_text"])
        total_latency  = round((time.time() - start_time) * 1000, 2)
        metric_result  = metrics.evaluate(
            original=raw_text,
            corrected=nlp_result["corrected_text"]
        )

        logger.log_full_pipeline({
            "raw_text":   raw_text,
            "nlp":        nlp_result,
            "braille":    braille_output,
            "latency_ms": total_latency
        })

        return jsonify({
            "recognized_text": raw_text,
            "corrected_text":  nlp_result["corrected_text"],
            "braille":         braille_output["unicode"],
            "braille_cells":   braille_output["cells"],
            "confidence":      confidence,
            "low_confidence":  nlp_result["low_confidence"],
            "wer":             metric_result["wer"],
            "latency_ms":      total_latency,
            "pipeline":        "complete"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── IMAGE → BRAILLE PIPELINE ───────────────────────────────────────────────────
@app.route("/api/image-to-braille", methods=["POST"])
def image_to_braille():
    """
    NEW ENDPOINT: Image → BLIP Caption + YOLO Detection → NLP → Braille

    Accepts: JSON body { "image": "<base64 string>" }
             where image is a base64-encoded JPEG/PNG from the webcam

    Returns:
        caption          → BLIP natural language description
        detected_objects → YOLO object list
        combined_text    → merged description
        corrected_text   → NLP-cleaned version
        braille          → Unicode Braille string
        braille_cells    → animated cell data
        latency_ms       → total pipeline time
    """
    try:
        import time
        start_time = time.time()

        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided. Send JSON with 'image' field."}), 400

        base64_image = data["image"]

        # ── Step 1: Image Analysis (BLIP + YOLO) ──────────────────
        img_result = imgproc.process_base64(base64_image)

        combined_text = img_result["combined_text"]

        # ── Step 2: NLP Processing ─────────────────────────────────
        nlp_result = nlp.process(combined_text, confidence=1.0)

        # ── Step 3: Braille Encoding ───────────────────────────────
        braille_output = encoder.encode(nlp_result["corrected_text"])

        total_latency = round((time.time() - start_time) * 1000, 2)

        return jsonify({
            "caption":          img_result["caption"],
            "detected_objects": img_result["detected_objects"],
            "object_scores":    img_result["object_scores"],
            "combined_text":    combined_text,
            "corrected_text":   nlp_result["corrected_text"],
            "braille":          braille_output["unicode"],
            "braille_cells":    braille_output["cells"],
            "latency_ms":       total_latency,
            "blip_available":   img_result["blip_available"],
            "yolo_available":   img_result["yolo_available"],
            "mock":             img_result["mock"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Run Server ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("   ViBroBraille Server Starting...")
    print("   Model: Whisper small + BLIP + YOLOv8")
    print("   Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
