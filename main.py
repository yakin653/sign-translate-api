"""
API FastAPI + session live (micro / Stereo Mix → chunks → Whisper → WebSocket → Angular).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from services.live_broadcaster import broadcaster
from services.sign_gloss import text_to_gloss_and_sequence
from services.url_media_transcribe import (
    download_youtube_audio_to_dir,
    fetch_youtube_transcript_text,
    is_allowed_youtube_url,
)
from services.whisper_runtime import transcribe_file_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sign-translate")

POSE_UPSTREAM = (
    "https://us-central1-sign-mt.cloudfunctions.net/spoken_text_to_signed_pose"
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    paths = [
        getattr(r, "path", "")
        for r in app.routes
        if "spoken_text_to_signed_pose" in getattr(r, "path", "") or "/pose/" in getattr(r, "path", "")
    ]
    logger.info("Routes pose actives : %s", paths)
    yield


app = FastAPI(
    title="Sign Translate API",
    description="Backend IA — HTTP + WebSocket live pour simulation PC.",
    version="0.3.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://0.0.0.0:4200",
        "http://localhost:4201",
        "http://127.0.0.1:4201",
    ],
    allow_origin_regex=(
        r".*"
        if os.environ.get("CORS_ALLOW_ALL", "").lower() in ("1", "true", "yes")
        else r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _frontend_dist_dir() -> Path | None:
    """
    Dossier du build Angular à servir en production.
    Par défaut : ../frontend/dist/sign-translate/browser
    Surchargable via FRONTEND_DIST_DIR.
    """
    raw = os.environ.get("FRONTEND_DIST_DIR", "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        return p if p.is_dir() else None
    # repo root = backend/.. ; then frontend/dist/...
    p = (Path(__file__).resolve().parent.parent / "frontend" / "dist" / "sign-translate" / "browser").resolve()
    return p if p.is_dir() else None


def _should_serve_frontend() -> bool:
    return os.environ.get("SERVE_FRONTEND", "").lower() in ("1", "true", "yes") and _frontend_dist_dir() is not None


_API_PREFIXES = (
    "/docs",
    "/openapi.json",
    "/redoc",
    "/sign-translate-ping",
    "/health",
    "/integration",
    "/ws/",
    "/live/",
    "/translate-text",
    "/spoken_text_to_signed_pose",
    "/pose/",
    "/api/",
)


async def proxy_spoken_to_signed_pose(text: str, spoken: str, signed: str) -> Response:
    """Relaie la pose sign.mt (GET). Même contrat que la Cloud Function (query : text, spoken, signed)."""
    upstream = (
        POSE_UPSTREAM
        + "?"
        + urllib.parse.urlencode({"text": text, "spoken": spoken, "signed": signed})
    )

    def fetch() -> tuple[bytes, str]:
        req = urllib.request.Request(upstream, method="GET")
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type") or "application/octet-stream"
            return body, ctype.split(";")[0].strip()

    try:
        body, media_type = await asyncio.to_thread(fetch)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:2000]
        logger.warning("Pose upstream HTTP %s : %s", e.code, detail[:200])
        raise HTTPException(status_code=e.code, detail=detail) from e
    except Exception as e:
        logger.exception("Pose upstream : %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e

    return Response(content=body, media_type=media_type)


# Même chemin que la Cloud Function (évite /pose vs /poses dans la barre d’adresse).
app.add_api_route(
    "/spoken_text_to_signed_pose",
    proxy_spoken_to_signed_pose,
    methods=["GET"],
    tags=["pose"],
)
app.add_api_route(
    "/pose/spoken-to-signed",
    proxy_spoken_to_signed_pose,
    methods=["GET"],
    tags=["pose"],
)
app.add_api_route(
    "/pose/spoken-to-signed/",
    proxy_spoken_to_signed_pose,
    methods=["GET"],
    tags=["pose"],
    include_in_schema=False,
)
app.add_api_route(
    "/api/v1/pose/spoken-to-signed",
    proxy_spoken_to_signed_pose,
    methods=["GET"],
    tags=["pose"],
    include_in_schema=False,
)


@app.get("/sign-translate-ping")
def sign_translate_ping() -> dict[str, str | dict[str, bool]]:
    """Pour scripts (dev-overlay) : détecte cette API même si un autre /health existe ailleurs."""
    return {
        "app": "sign-translate-api",
        "version": "0.3.0",
        "capabilities": {"transcribe_url": True, "transcribe_upload": True},
    }


@app.get("/health")
def health() -> dict[str, str | list[str]]:
    return {
        "status": "ok",
        "service": "sign-translate-api",
        "pose_proxy_paths": [
            "/spoken_text_to_signed_pose",
            "/pose/spoken-to-signed",
        ],
    }


@app.get("/integration")
def integration(request: Request) -> dict[str, str]:
    """
    Manifest pour clients embarqués (Flutter WebView, app native) : base HTTP et WebSocket.
    """
    base = str(request.base_url).rstrip("/")
    ws_origin = base.replace("http://", "ws://").replace("https://", "wss://")
    return {
        "version": "0.3.0",
        "api_base": base,
        "ws_live": f"{ws_origin}/ws/live",
        "phone_live": f"{base}/phone-live",
        "live_translate": f"{base}/live-translate",
        "whisper_lang": "POST /live/whisper-lang",
        "transcribe_upload": "POST /live/transcribe-upload",
    }


@app.get("/")
def root() -> dict[str, str]:
    # In production we serve the Angular build from the same domain.
    if _should_serve_frontend():
        dist = _frontend_dist_dir()
        assert dist is not None
        index = dist / "index.html"
        if not index.is_file():
            index = dist / "index.csr.html"
        if index.is_file():
            return FileResponse(index)
    return {
        "service": "sign-translate-api",
        "docs": "/docs",
        "health": "/health",
        "ping": "/sign-translate-ping",
        "ws_live": "/ws/live",
        "transcribe_upload": "POST /live/transcribe-upload (multipart: file, lang optionnel)",
        "transcribe_url": "POST /live/transcribe-url (JSON: url, lang optionnel — YouTube uniquement)",
        "pose_proxy": "/spoken_text_to_signed_pose?text=…&spoken=…&signed=…",
        "frontend": "SERVE_FRONTEND=1 pour servir Angular depuis FastAPI",
    }


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_fallback(full_path: str):
    """
    SPA fallback: sert les fichiers du build Angular et retombe sur index.html.
    Ne s'applique qu'avec SERVE_FRONTEND=1.
    """
    if not _should_serve_frontend():
        raise HTTPException(status_code=404, detail="Not Found")
    path_in = "/" + (full_path or "")
    if path_in.startswith(_API_PREFIXES):
        raise HTTPException(status_code=404, detail="Not Found")
    dist = _frontend_dist_dir()
    assert dist is not None
    # file resolve + sandbox inside dist
    candidate = (dist / full_path).resolve()
    try:
        candidate.relative_to(dist)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not Found")
    if candidate.is_file():
        return FileResponse(candidate)
    index = dist / "index.html"
    if not index.is_file():
        index = dist / "index.csr.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=500, detail="Frontend dist missing index")


class LiveWhisperLangBody(BaseModel):
    """Langue cible pour Whisper sur le flux live (alignée sur la langue parlée de l’UI)."""

    lang: str = Field(..., min_length=2, max_length=32, description="ex. fr, en, de")


@app.post("/live/whisper-lang")
def post_live_whisper_lang(body: LiveWhisperLangBody) -> dict[str, str | bool]:
    broadcaster.set_whisper_lang(body.lang.strip())
    return {"ok": True, "lang": broadcaster.effective_whisper_lang()}


@app.get("/live/status")
def live_status() -> dict:
    """État de la session live (nombre de clients WebSocket connectés)."""
    return {
        "clients": broadcaster.client_count,
        "chunk_sec": float(__import__("os").environ.get("LIVE_CHUNK_SEC", "2.5")),
    }


@app.get("/live/audio-devices")
def live_audio_devices() -> dict:
    """
    Liste des entrées audio (indices PortAudio). Utile pour régler AUDIO_DEVICE=…
    si la boucle loopback n'est pas détectée automatiquement.
    """
    from services.audio_chunk import (
        list_input_devices,
        resolve_input_device,
        soundcard_status,
        suggest_system_audio_device,
    )

    return {
        "resolved_input_device": resolve_input_device(),
        "suggested_auto_device": suggest_system_audio_device(),
        "soundcard": soundcard_status(),
        "inputs": list_input_devices(),
        "hint": "Sous Windows le son YouTube est capté via le paquet « soundcard » (loopback WASAPI). "
        "pip install -r requirements.txt puis redémarrer. AUDIO_CAPTURE=portaudio pour forcer PortAudio.",
    }


class TranslateTextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte source à traiter")


class TranslateTextResponse(BaseModel):
    text: str
    gloss: str
    sign_sequence: list[str] = Field(default_factory=list)


_MEDIA_UPLOAD_MAX_BYTES = int(os.environ.get("TRANSCRIBE_UPLOAD_MAX_MB", "120")) * 1024 * 1024
_TRANSCRIBE_POSE_MAX_CHARS = int(os.environ.get("TRANSCRIBE_POSE_MAX_CHARS", "260"))
_FORCE_FALLBACK = os.environ.get("TRANSCRIBE_FORCE_FALLBACK", "").lower() in ("1", "true", "yes")
_FALLBACK_TEXT = os.environ.get(
    "TRANSCRIBE_FALLBACK_TEXT",
    "Hello, this is a fallback transcript. The translation pipeline is active.",
).strip()
_MEDIA_ALLOWED_EXT = frozenset(
    {
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".webm",
        ".ogg",
        ".opus",
        ".flac",
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
    }
)


def _condense_for_pose(text: str, hard_limit: int = _TRANSCRIBE_POSE_MAX_CHARS) -> str:
    """
    Réduit la transcription à une taille robuste pour l'endpoint pose.
    Le service upstream renvoie souvent 503 sur des textes trop longs.
    """
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) <= hard_limit:
        return t
    short = t[:hard_limit]
    cut = max(short.rfind("."), short.rfind("!"), short.rfind("?"), short.rfind(","), short.rfind(" "))
    if cut >= int(hard_limit * 0.5):
        return short[: cut + 1].strip()
    return short.strip()


@app.post("/live/transcribe-upload")
async def transcribe_upload(
    file: UploadFile = File(...),
    lang: str = Form(""),
) -> dict[str, str]:
    """
    Transcrit un fichier audio ou vidéo (Whisper + ffmpeg), pour remplir le texte source côté Angular.
    """
    if _FORCE_FALLBACK:
        return {"text": _condense_for_pose(_FALLBACK_TEXT)}

    raw_name = file.filename or "upload.bin"
    suffix = Path(raw_name).suffix.lower()
    if suffix not in _MEDIA_ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Extension non supportée ({suffix or '—'}). "
            f"Formats : {', '.join(sorted(_MEDIA_ALLOWED_EXT))}",
        )

    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        total = 0
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MEDIA_UPLOAD_MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Fichier trop volumineux (max { _MEDIA_UPLOAD_MAX_BYTES // (1024 * 1024) } Mo).",
                    )
                out.write(chunk)

        lang_arg = lang.strip() or None
        try:
            text = await asyncio.to_thread(transcribe_file_path, tmp_path, lang_arg)
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            logger.exception("transcribe_upload: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

        text_out = _condense_for_pose(text)
        return {"text": text_out}
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class TranscribeUrlBody(BaseModel):
    """URL vidéo (liste blanche : YouTube seulement)."""

    url: str = Field(..., min_length=10, max_length=2000)
    lang: str = Field("", max_length=32)


@app.post("/live/transcribe-url")
async def transcribe_url(body: TranscribeUrlBody) -> dict[str, str]:
    """
    Télécharge l’audio via yt-dlp (YouTube), transcrit avec Whisper, renvoie le texte pour l’UI.
    """
    if _FORCE_FALLBACK:
        return {"text": _condense_for_pose(_FALLBACK_TEXT)}

    raw_url = body.url.strip()
    if not is_allowed_youtube_url(raw_url):
        raise HTTPException(
            status_code=400,
            detail="URL non autorisée ou invalide. Domaines acceptés : youtube.com, youtu.be, "
            "music.youtube.com, youtube-nocookie.com.",
        )

    tmpdir = tempfile.mkdtemp(prefix="sign-transcribe-url-")
    audio_path: str | None = None
    try:
        # Chemin rapide: sous-titres YouTube quand disponibles.
        lang_arg = body.lang.strip() or None
        fast_text = await asyncio.to_thread(fetch_youtube_transcript_text, raw_url, lang_arg)
        if fast_text:
            return {"text": _condense_for_pose(fast_text)}

        audio_path = await asyncio.to_thread(download_youtube_audio_to_dir, raw_url, tmpdir)
        try:
            text = await asyncio.to_thread(transcribe_file_path, audio_path, lang_arg)
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            logger.exception("transcribe_url whisper: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

        text_out = _condense_for_pose(text)
        return {"text": text_out}
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("transcribe_url: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post("/translate-text", response_model=TranslateTextResponse)
def translate_text(body: TranslateTextRequest) -> TranslateTextResponse:
    raw = body.text.strip()
    gloss, seq = text_to_gloss_and_sequence(raw)
    return TranslateTextResponse(
        text=raw,
        gloss=gloss,
        sign_sequence=seq,
    )


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    await websocket.accept()
    broadcaster.register(websocket)
    logger.info("WebSocket client connecté (%d actifs)", broadcaster.client_count)
    await broadcaster.ensure_running()
    try:
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.disconnect":
                break
            text = raw.get("text")
            if text is None:
                continue
            t = text.strip()
            if t.lower() in ("ping", '"ping"'):
                await websocket.send_text(json.dumps({"type": "pong", "ts": __import__("time").time()}))
                continue
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                continue
            mtype = body.get("type")
            if mtype == "live_source":
                mode = str(body.get("mode", "server")).lower()
                await broadcaster.set_live_source_mode("browser" if mode == "browser" else "server")
            elif mtype == "audio_pcm":
                b64 = body.get("data")
                sr = int(body.get("sampleRate", 16000) or 16000)
                if isinstance(b64, str) and b64:
                    await broadcaster.feed_browser_pcm_b64(b64, sr)
    except WebSocketDisconnect:
        logger.info("WebSocket déconnecté")
    finally:
        broadcaster.unregister(websocket)
        logger.info("Clients restants: %d", broadcaster.client_count)
