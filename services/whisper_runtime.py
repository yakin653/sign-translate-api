"""
Chargement paresseux de Whisper + transcription d'un chunk numpy float32 mono 16 kHz.
Variable d'environnement WHISPER_MOCK=1 pour tester sans modèle lourd.
"""

from __future__ import annotations

import logging
import os
import shutil
from subprocess import CalledProcessError, run
from typing import Any

import numpy as np

log = logging.getLogger("sign-translate.whisper")

_model: Any = None
_mock_i = 0
_ffmpeg_cmd: str | None = None
_MIN_TRANSCRIBE_SAMPLES = 1600  # 0.1 s à 16 kHz


def _mock_transcribe() -> str:
    global _mock_i
    samples = [
        "Bonjour, ceci est une simulation sans Whisper.",
        "Le microphone capte l'audio de votre PC.",
        "Activez Stereo Mix pour capter YouTube au lieu du micro.",
    ]
    t = samples[_mock_i % len(samples)]
    _mock_i += 1
    return t


def get_model_name() -> str:
    return os.environ.get("WHISPER_MODEL", "base")


def ensure_ffmpeg_available() -> str:
    """
    Garantit un binaire ffmpeg accessible.
    - 1) utilise ffmpeg déjà présent dans le PATH
    - 2) sinon, tente imageio-ffmpeg (binaire embarqué Python)
    """
    global _ffmpeg_cmd
    if _ffmpeg_cmd:
        return _ffmpeg_cmd
    existing = shutil.which("ffmpeg")
    if existing:
        _ffmpeg_cmd = existing
        return _ffmpeg_cmd
    try:
        import imageio_ffmpeg  # type: ignore

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()  # ex: ...\ffmpeg-win-x86_64-v7.1.exe
    except Exception as e:
        raise RuntimeError(
            "ffmpeg introuvable. Installez ffmpeg (ou pip install imageio-ffmpeg) puis redémarrez le backend."
        ) from e
    if not os.path.isfile(ffmpeg_exe):
        raise RuntimeError("ffmpeg fallback indisponible (imageio-ffmpeg).")
    _ffmpeg_cmd = ffmpeg_exe
    return _ffmpeg_cmd


def _load_audio_with_ffmpeg(
    file: str, ffmpeg_cmd: str, sr: int = 16000, max_seconds: float | None = None
) -> np.ndarray:
    cmd = [
        ffmpeg_cmd,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
    ]
    if max_seconds and max_seconds > 0:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += [
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        err = (e.stderr or b"").decode(errors="replace")
        raise RuntimeError(f"Failed to load audio: {err}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_model_if_needed() -> Any:
    global _model
    if os.environ.get("WHISPER_MOCK", "").lower() in ("1", "true", "yes"):
        return None
    if _model is None:
        import whisper

        name = get_model_name()
        log.info("Chargement Whisper (%s)…", name)
        _model = whisper.load_model(name)
        log.info("Whisper prêt.")
    return _model


def transcribe_file_path(path: str, language: str | None = None) -> str:
    """
    Transcription d’un fichier audio ou vidéo (ffmpeg via whisper.load_audio).
    Nécessite ffmpeg dans le PATH (installé avec openai-whisper en pratique).
    """
    if os.environ.get("WHISPER_MOCK", "").lower() in ("1", "true", "yes"):
        return _mock_transcribe()

    model = load_model_if_needed()
    if model is None:
        return _mock_transcribe()

    try:
        ffmpeg_cmd = ensure_ffmpeg_available()
        max_seconds_raw = os.environ.get("TRANSCRIBE_MAX_AUDIO_SECONDS", "120").strip()
        max_seconds = float(max_seconds_raw) if max_seconds_raw else 120.0
        audio = _load_audio_with_ffmpeg(path, ffmpeg_cmd, sr=16000, max_seconds=max_seconds)
        if audio.size == 0:
            raise RuntimeError("Aucun audio détecté dans le média (segment vide ou non supporté).")
    except Exception as e:
        log.exception("load_audio: %s", e)
        raise RuntimeError(
            f"Impossible de lire le média: {e}"
        ) from e

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.size < _MIN_TRANSCRIBE_SAMPLES:
        raise RuntimeError(
            "Audio trop court ou quasi silencieux pour transcription (minimum ~0.1 s utile)."
        )
    kwargs: dict = {"fp16": False}
    if language and language.strip() and language.lower() != "auto":
        kwargs["language"] = language.strip()
    try:
        result = model.transcribe(audio, **kwargs)
    except Exception as e:
        msg = str(e)
        if "cannot reshape tensor of 0 elements" in msg:
            raise RuntimeError(
                "Audio détecté mais inexploitable par Whisper (segment vide/silencieux). Réessayez avec un passage plus parlant."
            ) from e
        raise
    return (result.get("text") or "").strip()


def transcribe_chunk(audio: np.ndarray, language: str | None = None) -> str:
    if os.environ.get("WHISPER_MOCK", "").lower() in ("1", "true", "yes"):
        return _mock_transcribe()

    model = load_model_if_needed()
    if model is None:
        return _mock_transcribe()

    # whisper attend float32 numpy, 16 kHz
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.size < _MIN_TRANSCRIBE_SAMPLES:
        return ""
    kwargs: dict = {"fp16": False}
    if language and language.lower() != "auto":
        kwargs["language"] = language
    try:
        result = model.transcribe(audio, **kwargs)
    except Exception as e:
        # Évite de casser la boucle live sur segments trop courts/silencieux.
        if "cannot reshape tensor of 0 elements" in str(e):
            return ""
        raise
    return (result.get("text") or "").strip()
