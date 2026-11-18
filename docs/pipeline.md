# Pipeline (DVC)

```
raw files ──> clean ──> dataset ──> features ──┬──> train          -> models/embassy_latest
                                               └──> walk_forward   -> reports/walkforward_embassy.csv
```

## Daily commands

```bash
dvc dag            # draw the stages
dvc status         # what is out of date?
dvc repro          # run only what is out of date
dvc repro train    # one stage (and what it needs)
```

## First-time setup (once)

```bash
pip install -e ".[pipeline]"
dvc init                       # creates .dvc/ and .dvcignore
dvc add data/raw               # raw data -> DVC cache; creates data/raw.dvc
git add .dvc .dvcignore data/raw.dvc data/.gitignore dvc.yaml
```

After the first `dvc repro`, commit `dvc.lock` too. It is the record of exactly which data and code made which result.

## Backup of the data (optional, once)

A copy outside the project folder. `--local` keeps this machine-specific path out of git:

```bash
dvc remote add -d --local backup ~/dvc-backup
dvc push           # copy cached data and outputs to the backup
dvc pull           # on a new machine: get them back
```

## What is NOT a stage, and why

| Step | Why not |
|---|---|
| data download | needs API keys; results are cached already |
| frozen model + locked final test | must stay a deliberate manual step (`--final`) |
| sensor transfer test | one-off analysis |
| live batch forecast | runs on a schedule, not when data files change |

## Rules

- Never edit files in `data/processed/` or `models/embassy_latest` by hand. `dvc repro` overwrites them.
- A stage reruns when a listed dep changes. If you add an import to a stage's code, add the file to its `deps`.
- Commit `dvc.yaml` and `dvc.lock` together with the code change that caused them.
