# Running the whole system (Docker Compose)

```
              +------------------+        +--------------------+
OpenAQ  --->  |   forecaster     | -----> |  db (PostgreSQL)   | <----- api  <----- you (browser, curl)
(sensors)     |  every hour :10  |        |  forecasts, runs   |        read-only
              +------------------+        +--------------------+
                 |  uses models/embassy_latest (mounted, read-only)
                 +-> also appends data/forecasts/forecasts.csv (backup)
```

## First time

```bash
python docker/make_constraints.py          # pins the library versions to those of the trained model
docker compose build                       # builds the image (a few minutes the first time)
docker compose up -d --wait                # starts db, api, forecaster; waits until healthy
```

Needs `OPENAQ_API_KEY=...` in `.env` (compose reads it; it is never put into the image).
Stop any `uvicorn` you started by hand first: the api container uses the same port 8000.

## Every day

```bash
docker compose ps                          # all three should be running / healthy
docker compose logs -f forecaster          # follow the hourly runs (Ctrl+C stops following, not the service)
curl -s localhost:8000/health              # status ok / stale, last run, skipped sensors
docker compose down                        # stop (the database keeps its data)
```

## After retraining a model

```bash
dvc repro                                  # new models/embassy_latest
python docker/make_constraints.py          # versions may have changed
docker compose build
docker compose up -d --wait                # recreates api and forecaster with the new image
```

The new model has a new `model_version` label (`embassy_to_<last training date>`), so old and new forecasts stay separate in the database.

## If something is wrong

| Symptom | Look at |
|---|---|
| `/health` says `stale` | `docker compose logs forecaster`: is the loop running? API key set? OpenAQ down? |
| `/health` says `db_down` | `docker compose ps`: is `db` healthy? |
| forecaster says "No model found" | is `models/embassy_latest` there? run `dvc repro` |
| permission denied in `data/forecasts` | put `HAWA_UID` / `HAWA_GID` (output of `id -u`, `id -g`) in `.env` |
| `WARNING library version differs` | run `python docker/make_constraints.py` and rebuild |
| port 5433 or 8000 already used | stop the other program (a hand-started `uvicorn`, another Postgres) |

## What is NOT in the image, on purpose

Data, models and secrets. The image is only code and libraries, so it is small, can be shared safely,
and the same image runs with any model that you mount.
