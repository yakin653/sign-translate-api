# Backend — API IA (FastAPI)

Point d’entrée minimal pour intégrer plus tard Whisper, MediaPipe ou un pipeline de glosses côté serveur.

## Démarrage

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

- Ping (identifiant unique) : `GET http://localhost:8001/sign-translate-ping`
- Santé : `GET http://localhost:8001/health` (champ `service` = `sign-translate-api`)
- Swagger : `http://localhost:8001/docs`

## Prochaines étapes (suggestion)

- `POST /api/transcribe` : upload audio → texte (Whisper)
- `POST /api/sign-sequence` : texte → glosses / métadonnées pour le frontend Angular

Le frontend pointe vers l’API via `environment*.ts` (`localApiBaseUrl`, port **8001** par défaut).
