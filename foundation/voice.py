"""Voice-container inspection helpers; raw audio is never retained."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioInspection:
    container: str
    duration_seconds: float | None
    duration_verified: bool
    codec: str | None = None
    codec_verified: bool = False


def inspect_wav_duration(audio_bytes: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as source:
            rate = source.getframerate()
            frames = source.getnframes()
            return round(frames / rate, 3) if rate > 0 else None
    except (wave.Error, EOFError):
        return None


def _inspect_with_ffprobe(container: str, audio_bytes: bytes, executable: str) -> AudioInspection | None:
    suffix = ".wav" if container == "audio/wav" else ".audio"
    path = ""
    try:
        with tempfile.NamedTemporaryFile(prefix="gaash-voice-", suffix=suffix, delete=False) as handle:
            handle.write(audio_bytes)
            path = handle.name
        result = subprocess.run(
            [executable, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name,duration", "-of", "json", path],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        streams = json.loads(result.stdout).get("streams") or []
        stream = streams[0] if streams and isinstance(streams[0], dict) else {}
        codec = str(stream.get("codec_name") or "").strip().lower() or None
        try:
            duration = float(stream.get("duration"))
        except (TypeError, ValueError):
            duration = None
        return AudioInspection(
            container=container,
            duration_seconds=round(duration, 3) if duration is not None and duration >= 0 else None,
            duration_verified=duration is not None and duration >= 0,
            codec=codec,
            codec_verified=codec is not None,
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def inspect_audio(container: str, audio_bytes: bytes, ffprobe_path: str = "") -> AudioInspection:
    """Inspect only codecs that can be safely parsed without persistence.

    Other browser containers require the configured deployment validator (for
    example ffprobe); callers must expose this as unavailable rather than
    guessing a duration from file size.
    """

    if ffprobe_path:
        inspected = _inspect_with_ffprobe(container, audio_bytes, ffprobe_path)
        if inspected:
            return inspected
    if container == "audio/wav":
        duration = inspect_wav_duration(audio_bytes)
        return AudioInspection(container=container, duration_seconds=duration, duration_verified=duration is not None, codec="pcm", codec_verified=True)
    return AudioInspection(container=container, duration_seconds=None, duration_verified=False, codec=None, codec_verified=False)
