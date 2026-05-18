"""
==============================================================
  ViBroBraille - Image Processor Module
==============================================================

WHAT THIS DOES:
  1. BLIP Captioning  → generates a natural language description
                         of what's in the webcam image
  2. YOLOv8 Detection → detects specific objects (chair, cup,
                         person, door, etc.) with confidence scores
  3. Combined output  → merges caption + detected objects into
                         a single sentence → passed to Braille encoder

INSTALL:
    pip install transformers Pillow torch torchvision
    pip install ultralytics

FIRST RUN:
    BLIP model downloads ~900MB once → cached in ~/.cache/huggingface/
    YOLOv8n model downloads ~6MB once → cached in current directory

HOW IT WORKS:
    Browser captures webcam frame as base64 JPEG
          ↓
    Flask receives it → decodes to PIL Image
          ↓
    BLIP generates caption: "a person sitting at a desk"
          ↓
    YOLOv8 detects objects: ["person", "laptop", "cup"]
          ↓
    Combined: "A person sitting at a desk. Objects detected: person, laptop, cup."
          ↓
    NLP processor → Braille encoder → animated output
"""

import base64
import io
import time
import traceback
from PIL import Image


class ImageProcessor:
    """
    Handles image captioning (BLIP) and object detection (YOLOv8).

    Usage:
        processor = ImageProcessor()
        result    = processor.process_base64(base64_string)
        print(result["caption"])          # "a person near a door"
        print(result["detected_objects"]) # ["person", "door"]
        print(result["combined_text"])    # ready for Braille encoding
    """

    def __init__(self):
        self.blip_model     = None
        self.blip_processor = None
        self.yolo_model     = None
        self._load_blip()
        self._load_yolo()

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _load_blip(self):
        """
        Load BLIP image captioning model from HuggingFace.

        BLIP (Bootstrapped Language-Image Pretraining) generates
        natural language captions from images.

        Model: Salesforce/blip-image-captioning-base (~900MB, downloads once)
        """
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch

            print("[IMG] Loading BLIP captioning model...")
            model_name = "Salesforce/blip-image-captioning-base"

            self.blip_processor = BlipProcessor.from_pretrained(model_name)
            self.blip_model     = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            self.blip_model.eval()
            print("[IMG] ✓ BLIP model ready.")

        except ImportError:
            print("[IMG] ✗ transformers not installed.")
            print("[IMG]   Run: pip install transformers torch torchvision Pillow")
            self.blip_model = None
        except Exception as e:
            print(f"[IMG] ✗ BLIP load error: {e}")
            self.blip_model = None

    def _load_yolo(self):
        """
        Load YOLOv8 nano model for object detection.

        YOLOv8n is the smallest/fastest YOLOv8 variant (~6MB).
        It detects 80 common object classes (COCO dataset).

        Common detectable objects:
            person, bicycle, car, motorcycle, bus, train,
            chair, couch, bed, dining table, toilet, laptop,
            mouse, keyboard, phone, cup, bottle, book, clock,
            door (via wall/building detection), bag, umbrella...
        """
        try:
            from ultralytics import YOLO
            print("[IMG] Loading YOLOv8n object detection model...")
            self.yolo_model = YOLO("yolov8n.pt")  # downloads ~6MB on first run
            print("[IMG] ✓ YOLOv8 ready.")
        except ImportError:
            print("[IMG] ✗ ultralytics not installed.")
            print("[IMG]   Run: pip install ultralytics")
            self.yolo_model = None
        except Exception as e:
            print(f"[IMG] ✗ YOLO load error: {e}")
            self.yolo_model = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_base64(self, base64_image: str) -> dict:
        """
        Main entry point. Accepts a base64-encoded image string
        (as sent from browser webcam) and returns full analysis.

        Args:
            base64_image: Base64 string, optionally with data URI prefix
                          e.g. "data:image/jpeg;base64,/9j/4AAQ..."
                          or just the raw base64 string

        Returns:
            dict:
                caption          → BLIP natural language description
                detected_objects → list of object names from YOLO
                object_scores    → list of (object, confidence) tuples
                combined_text    → merged text ready for Braille encoding
                latency_ms       → total processing time
                blip_available   → True if BLIP model is loaded
                yolo_available   → True if YOLO model is loaded
                mock             → True if using fallback
        """
        start = time.time()

        try:
            # Decode base64 → PIL Image
            image = self._decode_base64_image(base64_image)

            caption          = ""
            detected_objects = []
            object_scores    = []

            # ── BLIP Captioning ────────────────────────────────────
            if self.blip_model is not None:
                caption = self._run_blip(image)
            else:
                caption = "Image captioning model not loaded."

            # ── YOLO Object Detection ──────────────────────────────
            if self.yolo_model is not None:
                detected_objects, object_scores = self._run_yolo(image)

            # ── Combine into single readable sentence ──────────────
            combined_text = self._build_combined_text(caption, detected_objects)

            latency_ms = round((time.time() - start) * 1000, 2)

            print(f"[IMG] Caption:  {caption}")
            print(f"[IMG] Objects:  {detected_objects}")
            print(f"[IMG] Latency:  {latency_ms}ms")

            return {
                "caption":          caption,
                "detected_objects": detected_objects,
                "object_scores":    object_scores,
                "combined_text":    combined_text,
                "latency_ms":       latency_ms,
                "blip_available":   self.blip_model is not None,
                "yolo_available":   self.yolo_model is not None,
                "mock":             False
            }

        except Exception as e:
            traceback.print_exc()
            return self._mock_result(str(e), time.time() - start)

    def process_pil_image(self, image: Image.Image) -> dict:
        """
        Alternative entry point — accepts a PIL Image directly.
        Same return format as process_base64().
        """
        # Convert PIL to base64 then reuse main pipeline
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return self.process_base64(b64)

    # ── BLIP Captioning ────────────────────────────────────────────────────────

    def _run_blip(self, image: Image.Image) -> str:
        """
        Generate a natural language caption for the image using BLIP.

        HOW BLIP WORKS:
          1. Image is resized and normalized into pixel tensors
          2. Vision encoder (ViT) extracts visual features
          3. Text decoder generates caption tokens autoregressively
          4. Beam search finds the most likely caption sequence

        BLIP is conditioned with a prompt "a photograph of" to guide
        it toward descriptive captions rather than abstract ones.
        """
        try:
            import torch

            # Prepare image for BLIP
            # Conditional captioning: give BLIP a text prompt to guide output
            text   = "a photograph of"
            inputs = self.blip_processor(
                images=image,
                text=text,
                return_tensors="pt"
            )

            with torch.no_grad():
                output_ids = self.blip_model.generate(
                    **inputs,
                    max_new_tokens=50,    # max caption length
                    num_beams=4,          # beam search width (higher = better quality)
                    min_length=5,         # ensure meaningful output
                    repetition_penalty=1.3 # reduce repetition in caption
                )

            caption = self.blip_processor.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            # Clean up the caption
            caption = caption.replace("a photograph of", "").strip()
            if caption and not caption[0].isupper():
                caption = caption[0].upper() + caption[1:]

            return caption

        except Exception as e:
            print(f"[IMG] BLIP error: {e}")
            return "Unable to generate image caption."

    # ── YOLO Object Detection ──────────────────────────────────────────────────

    def _run_yolo(self, image: Image.Image) -> tuple:
        """
        Detect objects in the image using YOLOv8.

        HOW YOLO WORKS:
          1. Image divided into a grid
          2. Each grid cell predicts bounding boxes + class probabilities
          3. Non-maximum suppression removes duplicate detections
          4. Returns list of detected objects with confidence scores

        We filter to confidence > 0.4 to reduce false positives.
        Objects are deduplicated (e.g. 3 "person" detections → 1 "person").

        Returns:
            tuple: (object_names_list, scores_list)
        """
        try:
            CONFIDENCE_THRESHOLD = 0.40  # minimum confidence to include detection

            results = self.yolo_model(
                image,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            detected     = {}  # object_name → highest confidence score
            result       = results[0]

            for box in result.boxes:
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]

                # Keep highest confidence score per class
                if class_name not in detected or detected[class_name] < confidence:
                    detected[class_name] = confidence

            # Sort by confidence (highest first)
            sorted_objects = sorted(detected.items(), key=lambda x: x[1], reverse=True)

            object_names  = [obj for obj, _    in sorted_objects]
            object_scores = [(obj, round(score, 2)) for obj, score in sorted_objects]

            return object_names, object_scores

        except Exception as e:
            print(f"[IMG] YOLO error: {e}")
            return [], []

    # ── Text Combination ───────────────────────────────────────────────────────

    def _build_combined_text(self, caption: str, objects: list) -> str:
        """
        Merge BLIP caption and YOLO object list into a single
        readable sentence for Braille encoding.

        Examples:
            caption="Person sitting at desk", objects=["person","laptop","cup"]
            → "Person sitting at desk. Objects detected: laptop, cup."

            caption="", objects=["chair","bottle"]
            → "Objects detected: chair, bottle."

            caption="Empty room", objects=[]
            → "Empty room."
        """
        parts = []

        if caption:
            # Ensure caption ends with period
            cap = caption.strip()
            if not cap.endswith("."):
                cap += "."
            parts.append(cap)

        if objects:
            # Remove objects already mentioned in caption to avoid repetition
            caption_lower    = caption.lower()
            unique_objects   = [o for o in objects if o not in caption_lower]

            if unique_objects:
                obj_list = ", ".join(unique_objects[:8])  # max 8 objects
                parts.append(f"Objects detected: {obj_list}.")

        if not parts:
            return "No content detected in image."

        return " ".join(parts)

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _decode_base64_image(self, base64_str: str) -> Image.Image:
        """
        Decode a base64 image string to a PIL Image.
        Handles both raw base64 and data URI format.
        """
        # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,")
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_str)
        image       = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB (removes alpha channel if PNG, handles grayscale)
        image = image.convert("RGB")

        return image

    def _mock_result(self, error_msg: str = "", latency: float = 0) -> dict:
        """Fallback result when models are not loaded or processing fails."""
        return {
            "caption":          "A person is standing in a room.",
            "detected_objects": ["person", "chair", "laptop"],
            "object_scores":    [("person", 0.95), ("chair", 0.82), ("laptop", 0.74)],
            "combined_text":    "A person is standing in a room. Objects detected: chair, laptop.",
            "latency_ms":       round(latency * 1000, 2),
            "blip_available":   self.blip_model is not None,
            "yolo_available":   self.yolo_model is not None,
            "mock":             True,
            "error":            error_msg
        }

    def is_ready(self) -> dict:
        """Return status of both models."""
        return {
            "blip": self.blip_model is not None,
            "yolo": self.yolo_model is not None
        }
