"""Privacy-preserving visual emotion inference for GAASH.

This module deliberately handles only explicitly submitted still images.  It
does not open a camera, retain raw images, infer clinical risk, or persist
media.  A caller may persist a successful, high-confidence *derived* signal
through the existing emotion-records store when that is appropriate.

The Hugging Face provider is an external inference client, not an in-process
``transformers`` pipeline.  ``HF_EMOTION_MODEL`` therefore identifies the
remote model and no model weights are downloaded into the GAASH process.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional


logger = logging.getLogger("gaash.visual_emotion")

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})

# This is a deliberately conservative engineering default, not a clinical
# threshold.  Operators may calibrate it for their approved model after
# validation, without changing request handlers or the frontend.
DEFAULT_CONFIDENCE_THRESHOLD = 0.60


class EmotionDetectorError(Exception):
    """Safe, stable error information for visual emotion inference."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class VisualEmotionSettings:
    """Backend-only provider configuration.

    ``provider`` remains ``deepface`` by default to preserve the current
    deployed behavior until the Hugging Face model and privacy approval have
    both been deliberately configured.
    """

    provider: str = "deepface"
    enabled: bool = True
    timeout_seconds: float = 15.0
    max_concurrency: int = 1
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    deepface_detector_backend: str = "opencv"
    hf_enabled: bool = False
    hf_token: str = ""
    hf_model: str = ""
    hf_model_version: str = ""

    @classmethod
    def from_environment(cls) -> "VisualEmotionSettings":
        provider = os.environ.get("VISUAL_EMOTION_PROVIDER", "deepface").strip().lower()
        if provider not in {"deepface", "huggingface"}:
            provider = "deepface"

        def as_bool(name: str, default: bool) -> bool:
            return os.environ.get(name, str(default)).strip().lower() == "true"

        def bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
            try:
                return min(max(float(os.environ.get(name, default)), minimum), maximum)
            except (TypeError, ValueError):
                return default

        def bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
            try:
                return min(max(int(os.environ.get(name, default)), minimum), maximum)
            except (TypeError, ValueError):
                return default

        legacy_timeout_key = "HF_VISION_TIMEOUT_SECONDS" if provider == "huggingface" else "DEEPFACE_TIMEOUT_SECONDS"
        legacy_concurrency_key = "HF_VISION_MAX_CONCURRENCY" if provider == "huggingface" else "DEEPFACE_MAX_CONCURRENCY"
        legacy_timeout_default = 15.0
        legacy_concurrency_default = 2 if provider == "huggingface" else 1

        return cls(
            provider=provider,
            enabled=as_bool("VISUAL_EMOTION_ENABLED", as_bool("DEEPFACE_ENABLED", True)),
            timeout_seconds=bounded_float(
                "VISUAL_EMOTION_TIMEOUT_SECONDS",
                bounded_float(legacy_timeout_key, legacy_timeout_default, 1.0, 120.0),
                1.0,
                120.0,
            ),
            max_concurrency=bounded_int(
                "VISUAL_EMOTION_MAX_CONCURRENCY",
                bounded_int(legacy_concurrency_key, legacy_concurrency_default, 1, 4),
                1,
                4,
            ),
            confidence_threshold=bounded_float(
                "HF_VISION_CONFIDENCE_THRESHOLD",
                DEFAULT_CONFIDENCE_THRESHOLD,
                0.0,
                1.0,
            ),
            deepface_detector_backend=os.environ.get("DEEPFACE_DETECTOR_BACKEND", "opencv").strip() or "opencv",
            hf_enabled=as_bool("HF_VISION_ENABLED", False),
            hf_token=os.environ.get("HF_TOKEN", "").strip(),
            hf_model=(
                os.environ.get(
                    "HF_EMOTION_MODEL",
                    "HardlyHumans/Facial-expression-detection",
                ).strip()
            ),
            hf_model_version=os.environ.get("HF_EMOTION_MODEL_VERSION", "").strip(),
        )


@dataclass(frozen=True)
class VisualEmotionResult:
    source: str = "visual"
    primary: Optional[str] = None
    confidence: Optional[float] = None
    scores: Mapping[str, float] = field(default_factory=dict)
    status: str = "unavailable"
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    raw_model_label: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status in {"classified", "uncertain"}


def normalize_emotion_label(value: object) -> Optional[str]:
    """Map supported model labels to GAASH's stable, non-clinical labels once.

    Unknown labels are not guessed or elevated into a stronger label.  The raw
    label remains available for internal operator diagnostics only.
    """

    label = str(value or "").strip().lower()
    aliases = {
        "angry": "anger",
        "anger": "anger",
        "disgust": "disgust",
        "disgusted": "disgust",
        "fear": "fear",
        "fearful": "fear",
        "happy": "joy",
        "happiness": "joy",
        "joy": "joy",
        "neutral": "neutral",
        "sad": "sadness",
        "sadness": "sadness",
        "surprise": "surprise",
        "surprised": "surprise",
    }
    return aliases.get(label)


def _safe_result(
    code: str,
    message: str,
    *,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
) -> VisualEmotionResult:
    return VisualEmotionResult(
        status=code.lower(),
        error_code=code,
        error_message=message,
        model_name=model_name,
        model_version=model_version,
    )


class VisualEmotionDetector:
    """Reusable, bounded visual inference service.

    The HF client is created once during FastAPI startup and reused.  HF model
    inference stays outside the event loop.  DeepFace remains an optional
    compatibility provider while an operator validates an approved HF model.
    """

    def __init__(
        self,
        settings: Optional[VisualEmotionSettings] = None,
        hf_client_factory: Optional[Callable[[str], object]] = None,
    ):
        self.settings = settings or VisualEmotionSettings.from_environment()
        self._hf_client_factory = hf_client_factory
        self._hf_client: Optional[object] = None
        self._startup_error: Optional[EmotionDetectorError] = None
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrency)

    @property
    def provider(self) -> str:
        return self.settings.provider

    @property
    def model_name(self) -> Optional[str]:
        if self.settings.provider == "huggingface":
            return self.settings.hf_model or None
        return "DeepFace" if self.settings.provider == "deepface" else None

    @property
    def model_version(self) -> Optional[str]:
        return self.settings.hf_model_version or None

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def readiness_error(self) -> Optional[EmotionDetectorError]:
        """Provider readiness without exposing credentials or media details."""
        return self._startup_error

    async def startup(self) -> None:
        """Prepare reusable lightweight provider state without processing media."""
        if not self.enabled or self.provider != "huggingface" or self._hf_client is not None:
            return
        if not self.settings.hf_enabled:
            self._startup_error = EmotionDetectorError(
                "MODEL_NOT_READY",
                "Visual emotion analysis is not enabled for the configured provider.",
            )
            return
        if not self.settings.hf_token or not self.settings.hf_model:
            self._startup_error = EmotionDetectorError(
                "MODEL_NOT_READY",
                "Visual emotion analysis is not configured.",
            )
            return
        try:
            factory = self._hf_client_factory or self._default_hf_client_factory
            self._hf_client = factory(self.settings.hf_token)
            self._startup_error = None
        except ImportError:
            self._startup_error = EmotionDetectorError(
                "MODEL_NOT_READY",
                "Visual emotion analysis is temporarily unavailable.",
            )
        except Exception as exc:  # Do not surface provider implementation details.
            logger.warning("HF visual emotion client initialization failed: %s", type(exc).__name__)
            self._startup_error = EmotionDetectorError(
                "MODEL_NOT_READY",
                "Visual emotion analysis is temporarily unavailable.",
            )

    async def shutdown(self) -> None:
        self._hf_client = None

    @staticmethod
    def _default_hf_client_factory(token: str) -> object:
        from huggingface_hub import InferenceClient

        return InferenceClient(provider="hf-inference", api_key=token)

    @staticmethod
    def decode_and_validate_image(image_base64: str) -> bytes:
        payload = str(image_base64 or "").strip()
        if payload.startswith("data:"):
            _, separator, payload = payload.partition(",")
            if not separator:
                raise EmotionDetectorError("INVALID_INPUT", "The supplied image data is not valid.")
        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise EmotionDetectorError("INVALID_INPUT", "The supplied image data is not valid.") from exc
        if not image_bytes:
            raise EmotionDetectorError("INVALID_INPUT", "The supplied image is empty.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise EmotionDetectorError("FILE_TOO_LARGE", "This image is too large to analyze. Choose an image under 8 MB.")
        try:
            from PIL import Image, ImageFile, UnidentifiedImageError

            ImageFile.LOAD_TRUNCATED_IMAGES = False
            with Image.open(io.BytesIO(image_bytes)) as image:
                image_format = str(image.format or "").upper()
                if image_format not in SUPPORTED_IMAGE_FORMATS:
                    raise EmotionDetectorError("UNSUPPORTED_MEDIA", "Use a JPEG, PNG, or WebP image for visual emotion analysis.")
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise EmotionDetectorError("FILE_TOO_LARGE", "This image is too large to analyze. Choose a smaller image.")
                image.verify()
        except EmotionDetectorError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise EmotionDetectorError("INVALID_INPUT", "The supplied image is not readable.") from exc
        return image_bytes

    async def analyze_base64(self, image_base64: str) -> VisualEmotionResult:
        if not self.enabled:
            return _safe_result("MODEL_NOT_READY", "Visual emotion analysis is disabled.", model_name=self.model_name, model_version=self.model_version)
        if self._startup_error is not None:
            return _safe_result(self._startup_error.code, self._startup_error.message, model_name=self.model_name, model_version=self.model_version)
        try:
            image_bytes = self.decode_and_validate_image(image_base64)
        except EmotionDetectorError as exc:
            return _safe_result(exc.code, exc.message, model_name=self.model_name, model_version=self.model_version)

        try:
            async with self._semaphore:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._analyze_sync, image_bytes),
                    timeout=self.settings.timeout_seconds,
                )
        except TimeoutError:
            logger.warning("Visual emotion inference timed out for provider=%s", self.provider)
            return _safe_result("INFERENCE_TIMEOUT", "Visual emotion analysis took too long. Please try again.", model_name=self.model_name, model_version=self.model_version)
        except EmotionDetectorError as exc:
            return _safe_result(exc.code, exc.message, model_name=self.model_name, model_version=self.model_version)
        except Exception as exc:  # A visual signal must never take down chat/API.
            logger.warning("Visual emotion inference failed for provider=%s: %s", self.provider, type(exc).__name__)
            return _safe_result("SERVER_ERROR", "Visual emotion analysis is temporarily unavailable.", model_name=self.model_name, model_version=self.model_version)

    def _analyze_sync(self, image_bytes: bytes) -> VisualEmotionResult:
        if self.provider == "huggingface":
            return self._analyze_huggingface(image_bytes)
        if self.provider == "deepface":
            return self._analyze_deepface(image_bytes)
        raise EmotionDetectorError("MODEL_NOT_READY", "Visual emotion analysis is not configured.")

    def _result_from_scores(self, scores: Mapping[str, float]) -> VisualEmotionResult:
        cleaned = {
            str(label).strip(): round(min(max(float(score), 0.0), 1.0) * 100, 2)
            for label, score in scores.items()
            if str(label).strip()
        }
        if not cleaned:
            raise EmotionDetectorError("SERVER_ERROR", "The visual emotion service returned no usable result.")
        raw_label, percentage = max(cleaned.items(), key=lambda item: item[1])
        confidence = round(percentage / 100, 4)
        common = {
            "source": "visual",
            "confidence": confidence,
            "scores": cleaned,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "raw_model_label": raw_label,
        }
        if confidence < self.settings.confidence_threshold:
            return VisualEmotionResult(status="uncertain", **common)
        primary = normalize_emotion_label(raw_label)
        if primary is None:
            return VisualEmotionResult(
                status="unsupported_label",
                error_code="UNSUPPORTED_LABEL",
                error_message="The configured model returned an unsupported emotion label.",
                **common,
            )
        return VisualEmotionResult(primary=primary, status="classified", **common)

    def _analyze_huggingface(self, image_bytes: bytes) -> VisualEmotionResult:
        if self._hf_client is None:
            raise EmotionDetectorError("MODEL_NOT_READY", "Visual emotion analysis is not ready.")
        try:
            predictions = self._hf_client.image_classification(image_bytes, model=self.settings.hf_model)
        except Exception as exc:
            logger.warning("HF visual emotion inference failed: %s", type(exc).__name__)
            raise EmotionDetectorError("SERVER_ERROR", "Visual emotion analysis is temporarily unavailable.") from exc
        if not isinstance(predictions, (list, tuple)):
            raise EmotionDetectorError("SERVER_ERROR", "The visual emotion service returned an unreadable result.")
        scores = {}
        for item in predictions:
            label = item.get("label") if isinstance(item, Mapping) else getattr(item, "label", None)
            score = item.get("score") if isinstance(item, Mapping) else getattr(item, "score", None)
            label = str(label or "").strip()
            if not label or score is None:
                continue
            try:
                scores[label] = float(score)
            except (TypeError, ValueError):
                continue
        return self._result_from_scores(scores)

    def _analyze_deepface(self, image_bytes: bytes) -> VisualEmotionResult:
        try:
            import numpy as np
            from deepface import DeepFace
            from PIL import Image
        except ImportError as exc:
            raise EmotionDetectorError("MODEL_NOT_READY", "Visual emotion analysis is temporarily unavailable.") from exc
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                frame = np.array(image.convert("RGB"))[:, :, ::-1]
            response = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend=self.settings.deepface_detector_backend,
                enforce_detection=False,
                silent=True,
            )
        except Exception as exc:
            logger.warning("DeepFace visual emotion inference failed: %s", type(exc).__name__)
            raise EmotionDetectorError("SERVER_ERROR", "Visual emotion analysis is temporarily unavailable.") from exc
        if isinstance(response, list):
            if not response:
                raise EmotionDetectorError("INVALID_INPUT", "No expression could be analyzed in the supplied image.")
            response = response[0]
        if not isinstance(response, Mapping):
            raise EmotionDetectorError("SERVER_ERROR", "The visual emotion service returned an unreadable result.")
        raw_scores = response.get("emotion", {})
        if not isinstance(raw_scores, Mapping):
            raise EmotionDetectorError("SERVER_ERROR", "The visual emotion service returned no usable result.")
        scores = {str(label): float(value) / 100 for label, value in raw_scores.items()}
        return self._result_from_scores(scores)
