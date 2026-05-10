"""
Téléchargement audio depuis une URL YouTube (yt-dlp) pour transcription Whisper.
Liste blanche stricte des hôtes pour limiter le SSRF.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.parse
from pathlib import Path

log = logging.getLogger("sign-translate.url-media")

_ALLOWED_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
        "www.youtube-nocookie.com",
        "youtube-nocookie.com",
    }
)


def extract_youtube_video_id(url: str) -> str | None:
    try:
        p = urllib.parse.urlparse((url or "").strip())
    except ValueError:
        return None
    host = (p.hostname or "").lower().rstrip(".")
    if host in ("youtu.be",):
        vid = p.path.lstrip("/").split("/")[0]
        return vid or None
    if host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
        if p.path == "/watch":
            q = urllib.parse.parse_qs(p.query or "")
            v = (q.get("v") or [""])[0].strip()
            return v or None
        if p.path.startswith("/shorts/") or p.path.startswith("/embed/"):
            parts = [x for x in p.path.split("/") if x]
            if len(parts) >= 2:
                return parts[1]
    return None


def fetch_youtube_transcript_text(url: str, preferred_lang: str | None = None) -> str:
    """
    Récupère les sous-titres YouTube (auto ou manuels) sans ffmpeg.
    Retourne chaîne vide si indisponible.
    """
    vid = extract_youtube_video_id(url)
    if not vid:
        return ""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception:
        return ""
    lang = (preferred_lang or "").strip().lower()
    langs = [lang] if lang else []
    if "en" not in langs:
        langs.append("en")
    if "fr" not in langs:
        langs.append("fr")
    try:
        items = YouTubeTranscriptApi.get_transcript(vid, languages=langs or None)
    except Exception:
        return ""
    text = " ".join((x.get("text") or "").strip() for x in items if (x.get("text") or "").strip())
    return text.strip()


def is_allowed_youtube_url(url: str) -> bool:
    raw = (url or "").strip()
    if not raw or len(raw) > 2000:
        return False
    try:
        p = urllib.parse.urlparse(raw)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    h = (p.hostname or "").lower().rstrip(".")
    return h in _ALLOWED_HOSTS


def download_youtube_audio_to_dir(url: str, out_dir: str) -> str:
    """
    Télécharge la meilleure piste audio disponible dans ``out_dir``.
    Retourne le chemin du fichier audio créé.
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError(
            "Le paquet yt-dlp est requis pour les URLs. Installez-le : pip install yt-dlp"
        ) from e

    max_sec = int(os.environ.get("TRANSCRIBE_URL_MAX_SECONDS", "900"))

    def _match_filter(info: dict, *, incomplete: bool = False) -> str | None:
        if incomplete:
            return None
        dur = info.get("duration")
        if dur is not None and float(dur) > max_sec:
            return (
                f"Vidéo trop longue ({int(float(dur) // 60)} min). "
                f"Maximum autorisé : {max_sec // 60} minutes."
            )
        return None

    ffmpeg_location: str | None = shutil.which("ffmpeg")
    if not ffmpeg_location:
        try:
            import imageio_ffmpeg  # type: ignore

            ffmpeg_location = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_location = None

    outtmpl = str(Path(out_dir) / "sign_audio.%(ext)s")
    opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 90,
        "retries": 2,
        "noplaylist": True,
        "match_filter": _match_filter,
        # Mode compatible : pas de téléchargement partiel (certaines configs yt-dlp exigent ffmpeg en PATH strict).
    }
    if ffmpeg_location:
        opts["ffmpeg_location"] = ffmpeg_location

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            code = ydl.download([url])
        if code != 0:
            log.warning("yt-dlp code retour %s pour %s", code, url)
    except Exception as e:
        log.exception("yt-dlp: %s", e)
        raise RuntimeError(
            "Impossible de récupérer la vidéo (réseau, vidéo privée, ou service indisponible)."
        ) from e

    exts = {".m4a", ".webm", ".opus", ".mp3", ".ogg", ".wav", ".aac", ".mp4", ".mkv"}
    files = [f for f in Path(out_dir).iterdir() if f.is_file() and f.suffix.lower() in exts]
    if not files:
        raise RuntimeError(
            "Aucun fichier audio extrait. Vérifiez l'URL, ou installez / mettez à jour ffmpeg et yt-dlp."
        )
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return str(files[0])
