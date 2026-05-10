"""
Une session producteur : enregistre des chunks, transcrit, diffuse à tous les clients WebSocket.

Modes :
- **serveur** (défaut) : capture micro / loopback PC (soundcard / PortAudio).
- **browser** : PCM float32 mono 16 kHz envoyé par le navigateur (partage d’onglet / fenêtre avec audio).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any

import numpy as np

from .audio_chunk import is_mostly_silent, record_mono_chunk
from .sign_gloss import text_to_gloss_and_sequence
from .whisper_runtime import transcribe_chunk

log = logging.getLogger("sign-translate.live")

LANGUAGE = os.environ.get("WHISPER_LANG", "fr")
CHUNK_SEC = float(os.environ.get("LIVE_CHUNK_SEC", "2.5"))
TARGET_SR = 16000
MIN_TEXT_CHARS = int(os.environ.get("LIVE_MIN_TEXT_CHARS", "3"))
DEDUP_SEC = float(os.environ.get("LIVE_DEDUP_SEC", "6.0"))


def _norm_text(s: str) -> str:
    t = (s or "").strip().lower()
    # collapse whitespace + remove trivial punctuation that causes repeats
    t = " ".join(t.split())
    for ch in (".", ",", "!", "?", "…", ":", ";", "\"", "'"):
        t = t.replace(ch, "")
    return t.strip()


def _resample_mono(x: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return np.array([], dtype=np.float32)
    if orig_sr == target_sr:
        return x.astype(np.float32)
    ratio = target_sr / orig_sr
    n_out = max(1, int(len(x) * ratio))
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x).astype(np.float32)


class LiveBroadcaster:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._clients: set[Any] = set()
        self._ensure_lock = asyncio.Lock()
        self._pcm_buf_lock = asyncio.Lock()
        self._chunk_index = 0
        self._whisper_lang_override: str | None = None
        self._prefer_browser_tab = False
        self._browser_pcm = np.array([], dtype=np.float32)
        self._browser_idle_ticks = 0
        # Anti-spam / stabilité du live
        self._last_norm: str = ""
        self._last_emit_ts: float = 0.0

    def set_whisper_lang(self, code: str | None) -> None:
        c = (code or "").strip()
        self._whisper_lang_override = c if c else None
        log.info("Whisper lang effective=%s (override=%s)", self.effective_whisper_lang(), self._whisper_lang_override)

    def effective_whisper_lang(self) -> str:
        if self._whisper_lang_override:
            return self._whisper_lang_override
        return (LANGUAGE or "fr").strip() or "fr"

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def register(self, ws: Any) -> None:
        self._clients.add(ws)

    def unregister(self, ws: Any) -> None:
        self._clients.discard(ws)

    async def broadcast(self, payload: dict) -> None:
        dead: list[Any] = []
        text = json.dumps(payload, ensure_ascii=False)
        for ws in self._clients:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def set_live_source_mode(self, mode: str) -> None:
        m = (mode or "server").strip().lower()
        prefer = m == "browser"
        async with self._pcm_buf_lock:
            self._prefer_browser_tab = prefer
            if not prefer:
                self._browser_pcm = np.array([], dtype=np.float32)
        log.info("Source live = %s", "navigateur (onglet)" if prefer else "serveur (micro / loopback)")
        await self.broadcast(
            {
                "type": "status",
                "message": (
                    "Audio depuis le navigateur (onglet / fenêtre partagé)."
                    if prefer
                    else "Audio depuis le serveur (micro ou loopback PC)."
                ),
            }
        )

    async def feed_browser_pcm_b64(self, pcm_b64: str, sample_rate: int) -> None:
        raw = base64.b64decode(pcm_b64, validate=False)
        if len(raw) < 4 or len(raw) % 4 != 0:
            return
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        mono = _resample_mono(arr, int(sample_rate) if sample_rate > 0 else TARGET_SR, TARGET_SR)
        async with self._pcm_buf_lock:
            if not self._prefer_browser_tab:
                return
            self._browser_pcm = np.concatenate([self._browser_pcm, mono])
            # limite mémoire (~15 s à 16 kHz)
            max_keep = TARGET_SR * 15
            if self._browser_pcm.size > max_keep:
                self._browser_pcm = self._browser_pcm[-max_keep:]

    async def _next_chunk(self, need: int) -> np.ndarray | None:
        """Retourne un bloc mono 16 kHz de longueur `need`, ou None si mode navigateur et pas assez de données."""
        if self._prefer_browser_tab:
            async with self._pcm_buf_lock:
                if self._browser_pcm.size >= need:
                    chunk = self._browser_pcm[:need].copy()
                    self._browser_pcm = self._browser_pcm[need:]
                    return chunk
            return None
        return await asyncio.to_thread(record_mono_chunk, float(CHUNK_SEC), int(TARGET_SR))

    async def _loop(self) -> None:
        effective = self.effective_whisper_lang()
        lang = None if effective.lower() == "auto" else effective
        await self.broadcast(
            {
                "type": "status",
                "message": (
                    f"Session live — chunks {CHUNK_SEC}s, Whisper={effective}. "
                    "Par défaut : micro / loopback serveur. "
                    "Vous pouvez aussi partager un onglet avec audio depuis la page Traduire."
                ),
            }
        )
        silent_streak = 0
        need = int(CHUNK_SEC * TARGET_SR)
        while self.client_count > 0:
            try:
                audio = await self._next_chunk(need)
            except Exception as e:
                log.exception("Capture audio: %s", e)
                await self.broadcast({"type": "error", "message": str(e)})
                await asyncio.sleep(1.0)
                continue

            if audio is None:
                self._browser_idle_ticks += 1
                if self._browser_idle_ticks == 25:
                    await self.broadcast(
                        {
                            "type": "status",
                            "message": (
                                "Mode onglet : en attente d’audio… Lancez une vidéo ou de la musique "
                                "dans l’onglet / fenêtre que vous avez partagé(e)."
                            ),
                        }
                    )
                if self._browser_idle_ticks >= 25 and self._browser_idle_ticks % 100 == 0:
                    await self.broadcast(
                        {
                            "type": "status",
                            "message": "Toujours pas assez de signal depuis le navigateur — vérifiez le partage d’onglet avec audio.",
                        }
                    )
                await asyncio.sleep(0.05)
                continue

            self._browser_idle_ticks = 0

            if is_mostly_silent(audio):
                self._chunk_index += 1
                silent_streak += 1
                if silent_streak == 12 and not self._prefer_browser_tab:
                    await self.broadcast(
                        {
                            "type": "status",
                            "message": (
                                "Audio très faible ou silence — le son de YouTube ne va pas au micro. "
                                "Essayez « Partager l’audio d’un onglet » dans la page Traduire, ou Stereo Mix. "
                                f"Voir http://127.0.0.1:{os.environ.get('SIGN_TRANSLATE_API_PORT', '8001')}/live/audio-devices"
                            ),
                        }
                    )
                if silent_streak >= 12 and silent_streak % 24 == 0 and not self._prefer_browser_tab:
                    await self.broadcast(
                        {
                            "type": "status",
                            "message": f"Toujours silence (chunk {self._chunk_index}). Parlez ou partagez l’audio d’un onglet.",
                        }
                    )
                continue
            silent_streak = 0

            effective = self.effective_whisper_lang()
            lang = None if effective.lower() == "auto" else effective

            try:
                text = await asyncio.to_thread(transcribe_chunk, audio, lang)
            except Exception as e:
                log.exception("Whisper: %s", e)
                await self.broadcast({"type": "error", "message": str(e)})
                self._chunk_index += 1
                continue

            if not text:
                self._chunk_index += 1
                continue

            # Filtre pro: ignore micro-hallucinations (1-2 chars) + déduplication temporelle
            norm = _norm_text(text)
            now = time.time()
            if len(norm) < MIN_TEXT_CHARS:
                self._chunk_index += 1
                continue
            if norm and norm == self._last_norm and (now - self._last_emit_ts) < DEDUP_SEC:
                # drop repeat
                self._chunk_index += 1
                continue
            self._last_norm = norm
            self._last_emit_ts = now

            gloss, seq = text_to_gloss_and_sequence(text)
            self._chunk_index += 1
            await self.broadcast(
                {
                    "type": "segment",
                    "text": text,
                    "gloss": gloss,
                    "sign_sequence": seq,
                    "chunk_index": self._chunk_index,
                    "ts": time.time(),
                }
            )

        log.info("Session live arrêtée (plus de clients).")

    async def ensure_running(self) -> None:
        async with self._ensure_lock:
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._loop())


broadcaster = LiveBroadcaster()
