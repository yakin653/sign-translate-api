"""
Capture mono 16 kHz pour Whisper.

Sous Windows, la capture « son du PC / YouTube » utilise d’abord **soundcard** (WASAPI loopback),
puis PortAudio (sounddevice) en secours.

Variables utiles :
  AUDIO_CAPTURE=auto|soundcard|portaudio  (défaut : auto)
  AUDIO_PORTAUDIO_TAP=1  — autorise l’auto-sélection PortAudio (Wave Speaker, etc.)
  AUDIO_DEVICE=id        — force un index PortAudio
  AUDIO_SYSTEM_TAP=0       — désactive le scoring Wave Speaker pour PortAudio
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None  # type: ignore

try:
    import soundcard as sc_audio
except ImportError:
    sc_audio = None  # type: ignore

log = logging.getLogger("sign-translate.audio")

_device_cache: int | None | str = "unset"  # unset | None (defaut) | int
# Une fois un couple (device, sr) validé par un enregistrement réussi, on le réutilise.
_verified: tuple[int | None, float] | str = "unset"
# None = pas encore essayé, True = soundcard OK, False = utiliser PortAudio seulement
_use_soundcard_loopback: bool | None = None


def _parse_audio_device() -> int | None:
    raw = os.environ.get("AUDIO_DEVICE", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        log.warning("AUDIO_DEVICE invalide (%s), ignoré.", raw)
        return None


def _score_windows_system_capture(name: str) -> int:
    n = (name or "").lower()
    if "mappeur de sons" in n or "sound mapper" in n:
        return 0
    if "loopback" in n:
        return 100
    if (
        "stereo mix" in n
        or "mixage stéréo" in n
        or "mixage stereo" in n
        or "what u hear" in n
        or "what you hear" in n
    ):
        return 95
    if "audiominiport wave speaker" in n:
        return 78 if "headphone" in n else 82
    if "wave speaker" in n and "input" in n:
        return 72
    return 0


def _find_windows_system_audio_device() -> int | None:
    if sd is None:
        return None
    if os.environ.get("AUDIO_SYSTEM_TAP", "").lower() in ("0", "false", "no"):
        return None
    try:
        devices = sd.query_devices()
    except Exception as e:
        log.warning("query_devices: %s", e)
        return None
    best: tuple[int, int, str] | None = None
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        raw_name = d.get("name") or ""
        sc = _score_windows_system_capture(raw_name)
        if sc <= 0:
            continue
        if best is None or sc > best[0] or (sc == best[0] and i < best[1]):
            best = (sc, i, raw_name)
    if best is not None:
        log.info(
            "Candidat « son PC » (score=%s) : [%s] %s",
            best[0],
            best[1],
            best[2][:80],
        )
        return int(best[1])
    return None


def resolve_input_device() -> int | None:
    global _device_cache
    if _device_cache != "unset":  # type: ignore[comparison-overlap]
        return _device_cache  # type: ignore[return-value]

    forced = _parse_audio_device()
    if forced is not None:
        _device_cache = forced
        log.info("AUDIO_DEVICE=%s", forced)
        return forced

    use_loopback = os.environ.get("AUDIO_LOOPBACK", "").lower()
    if use_loopback in ("0", "false", "no"):
        _device_cache = None
        log.info("AUDIO_LOOPBACK désactivé — entrée par défaut.")
        return None

    # Par défaut on NE force plus les entrées PortAudio « Wave Speaker » (souvent PaError -9996).
    # La capture YouTube passe par soundcard (loopback WASAPI). Pour l’ancien comportement :
    # $env:AUDIO_PORTAUDIO_TAP = "1"
    if sys.platform == "win32" and os.environ.get("AUDIO_PORTAUDIO_TAP", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        dev = _find_windows_system_audio_device()
        if dev is not None:
            _device_cache = dev
            return dev
        log.warning(
            "Aucune entrée « son PC » PortAudio détectée. Utilisez soundcard (défaut) ou Stereo Mix."
        )

    _device_cache = None
    return None


def _scored_input_indices_desc() -> list[int]:
    """Indices candidats (score > 0), du meilleur au moins bon."""
    if sd is None:
        return []
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    scored: list[tuple[int, int]] = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        sc = _score_windows_system_capture(d.get("name") or "")
        if sc > 0:
            scored.append((sc, i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [i for _, i in scored]


def _candidate_devices() -> list[int | None]:
    """Ordre d’essai : préféré → autres scores → micro par défaut (None)."""
    pref = resolve_input_device()
    seen: set[int | None] = set()
    out: list[int | None] = []
    for d in [pref] + _scored_input_indices_desc() + [None]:
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


def _pick_loopback_microphone():
    """Microphone loopback WASAPI aligné sur la sortie par défaut (son du PC / YouTube)."""
    assert sc_audio is not None
    spk = sc_audio.default_speaker()
    name = str(spk.name)
    try:
        return sc_audio.get_microphone(id=name, include_loopback=True)
    except (IndexError, ValueError, OSError) as e:
        log.warning("soundcard get_microphone(%r): %s — recherche parmi les loopbacks…", name, e)
    for m in sc_audio.all_microphones(include_loopback=True):
        if getattr(m, "isloopback", False):
            return m
    raise RuntimeError("Aucun périphérique loopback (soundcard).")


def _record_soundcard_loopback(seconds: float, target_sr: float) -> np.ndarray:
    """Enregistre le mix sortie (YouTube, etc.) via soundcard / WASAPI."""
    assert sc_audio is not None
    mic = _pick_loopback_microphone()
    rec_sr = 48000.0
    numframes = max(1, int(seconds * rec_sr))
    # Windows WASAPI : 2 canaux évite souvent des données corrompues (doc soundcard).
    data = mic.record(numframes=numframes, samplerate=int(rec_sr), channels=2)
    if data.ndim == 2:
        mono = np.mean(data, axis=1).astype(np.float32)
    else:
        mono = np.asarray(data, dtype=np.float32).flatten()
    return _resample_linear(mono, rec_sr, target_sr)


def soundcard_status() -> dict[str, Any]:
    """Pour GET /live/audio-devices."""
    if sc_audio is None:
        return {"available": False, "reason": "pip install soundcard"}
    try:
        loopbacks = [
            {"name": m.name, "isloopback": getattr(m, "isloopback", False)}
            for m in sc_audio.all_microphones(include_loopback=True)
            if getattr(m, "isloopback", False)
        ]
        spk = sc_audio.default_speaker()
        return {
            "available": True,
            "default_speaker": str(spk.name),
            "loopbacks": loopbacks,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def _resample_linear(audio: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    duration = len(audio) / float(orig_sr)
    n_target = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, duration, num=n_target, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.float32)


def _try_record(
    device: int | None,
    n: int,
    target_sr: float,
) -> np.ndarray:
    """Enregistre n échantillons à target_sr (mono float32)."""
    assert sd is not None
    last: Exception | None = None
    for ch in (1, 2):
        try:
            sd.check_input_settings(
                device=device,
                samplerate=target_sr,
                channels=ch,
                dtype="float32",
            )
            a = sd.rec(
                n,
                samplerate=target_sr,
                channels=ch,
                dtype=np.float32,
                device=device,
            )
            sd.wait()
            arr = np.asarray(a, dtype=np.float32)
            if ch == 1:
                return arr.flatten()
            return arr[:, 0].flatten()
        except Exception as e:
            last = e
            continue
    if last:
        raise last
    raise RuntimeError("Enregistrement impossible (mono/stéréo).")


def _record_with_device_samplerates(
    device: int | None,
    n: int,
    target_sr: float,
) -> np.ndarray:
    """Essaie plusieurs fréquences d’échantillonnage (certaines entrées refusent 16 kHz)."""
    assert sd is not None
    rates: list[float] = [target_sr]
    if device is not None:
        try:
            info = sd.query_devices(device)
            ds = float(info.get("default_samplerate") or 0)
            if ds > 0 and ds not in rates:
                rates.append(ds)
        except Exception:
            pass
    for extra in (48000.0, 44100.0):
        if extra not in rates:
            rates.append(extra)

    last_err: Exception | None = None
    for sr in rates:
        try:
            n_i = int(round(n * sr / target_sr)) if sr != target_sr else n
            raw = _try_record(device, n_i, sr)
            if sr != target_sr:
                raw = _resample_linear(raw, sr, target_sr)
            if len(raw) >= n:
                return raw[:n]
            if len(raw) < n:
                pad = np.zeros(n - len(raw), dtype=np.float32)
                return np.concatenate([raw, pad])
            return raw
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("Aucune combinaison samplerate / canaux n’a fonctionné.")


def record_mono_chunk(seconds: float, samplerate: int = 16000) -> np.ndarray:
    global _verified, _use_soundcard_loopback
    n = int(seconds * samplerate)
    target_sr = float(samplerate)
    mode = os.environ.get("AUDIO_CAPTURE", "auto").lower()
    if mode not in ("auto", "soundcard", "portaudio"):
        mode = "auto"

    # 1) Windows : soundcard = vrai loopback WASAPI (YouTube sans Stereo Mix)
    if (
        mode in ("auto", "soundcard")
        and sc_audio is not None
        and sys.platform == "win32"
        and mode != "portaudio"
    ):
        if _use_soundcard_loopback is True:
            try:
                return _record_soundcard_loopback(seconds, target_sr)
            except Exception as e:
                log.warning("soundcard : %s — repli PortAudio", e)
                _use_soundcard_loopback = False
        if _use_soundcard_loopback is not False:
            try:
                out = _record_soundcard_loopback(seconds, target_sr)
                _use_soundcard_loopback = True
                log.info("Capture soundcard loopback OK.")
                return out
            except Exception as e:
                log.warning("soundcard loopback indisponible : %s — passage à PortAudio", e)
                _use_soundcard_loopback = False
                if mode == "soundcard":
                    raise

    if mode == "soundcard" and sc_audio is None:
        raise RuntimeError(
            "AUDIO_CAPTURE=soundcard mais le module soundcard n'est pas installé. "
            "pip install soundcard"
        )

    # 2) PortAudio / sounddevice
    if sd is None:
        raise RuntimeError(
            "sounddevice n'est pas installé. pip install sounddevice numpy"
        )

    if _verified != "unset" and isinstance(_verified, tuple) and len(_verified) == 2:
        dev, _sr_use = _verified
        try:
            return _record_with_device_samplerates(dev, n, target_sr)
        except Exception as e:
            log.warning("Périphérique validé ne répond plus (%s), nouvelle détection…", e)
            _verified = "unset"

    candidates = _candidate_devices()
    last_err: Exception | None = None
    for dev in candidates:
        try:
            out = _record_with_device_samplerates(dev, n, target_sr)
            _verified = (dev, target_sr)
            log.info(
                "Capture PortAudio OK — device=%s (None = entrée par défaut).",
                dev,
            )
            return out
        except Exception as e:
            last_err = e
            log.warning(
                "Impossible d’ouvrir l’entrée audio device=%s : %s",
                dev,
                e,
            )
            continue

    if last_err:
        raise last_err
    raise RuntimeError("Aucun périphérique d’entrée utilisable.")


def is_mostly_silent(audio: np.ndarray, threshold: float | None = None) -> bool:
    if audio.size == 0:
        return True
    thr = threshold if threshold is not None else float(os.environ.get("SILENCE_RMS", "0.002"))
    return float(np.abs(audio).mean()) < thr


def list_input_devices() -> list[dict[str, Any]]:
    if sd is None:
        return []
    out: list[dict[str, Any]] = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                nm = d.get("name") or ""
                nl = nm.lower()
                out.append(
                    {
                        "id": i,
                        "name": d.get("name"),
                        "channels": d.get("max_input_channels"),
                        "default_samplerate": d.get("default_samplerate"),
                        "loopback": "loopback" in nl,
                        "system_capture_score": _score_windows_system_capture(nm),
                    }
                )
    except Exception as e:
        return [{"error": str(e)}]
    return out


def suggest_system_audio_device() -> dict[str, Any] | None:
    if sd is None or sys.platform != "win32":
        return None
    try:
        devices = sd.query_devices()
    except Exception:
        return None
    best: tuple[int, int, str] | None = None
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        raw = d.get("name") or ""
        sc = _score_windows_system_capture(raw)
        if sc <= 0:
            continue
        if best is None or sc > best[0] or (sc == best[0] and i < best[1]):
            best = (sc, i, raw)
    if best is None:
        return None
    return {"id": best[1], "name": best[2], "score": best[0]}

