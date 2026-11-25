"""Run the batch forecast every hour, forever. This is what the `forecaster` container runs.

    python -m hawa.forecasting.loop --model models/embassy_latest

It runs once at start, then at minute 10 of every hour (OpenAQ needs a few minutes to publish the hour that just
ended). If one run fails (network down, no data), the error is printed and the loop tries again next hour.
Every run saves to the CSV and to the database (`batch --to-db`), then compares the last 7 days of forecasts with
the real values (`actuals --to-db`). The keys make repeated runs safe.
"""
import argparse
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

from hawa.forecasting import actuals, batch

RUN_MINUTE = 10


def seconds_until_next(now, minute=RUN_MINUTE):
    """Seconds from `now` to the next time the clock shows HH:<minute>:00 (UTC)."""
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    target = now.floor("h") + pd.Timedelta(minutes=minute)
    if target <= now:
        target += pd.Timedelta(hours=1)
    return (target - now).total_seconds()


def run_forever(job, minute=RUN_MINUTE, clock=lambda: datetime.now(timezone.utc), sleep=time.sleep, max_runs=None):
    """Call `job()` now and then every hour. A failing job never stops the loop. `max_runs` is for tests."""
    runs = 0
    while True:
        started = clock()
        print(f"--- run at {pd.Timestamp(started):%Y-%m-%d %H:%M} UTC", flush=True)
        try:
            job()
        except (Exception, SystemExit) as err:      # SystemExit: batch.main exits when no model is found
            print(f"run failed: {type(err).__name__}: {err}", flush=True)
            traceback.print_exc()
        runs += 1
        if max_runs is not None and runs >= max_runs:
            return
        wait = seconds_until_next(clock(), minute)
        print(f"next run in {wait / 60:.0f} min", flush=True)
        sleep(wait)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="bundle folder (default: newest bundle trained on all data)")
    p.add_argument("--minute", type=int, default=RUN_MINUTE, help="minute of the hour to run (default 10)")
    a = p.parse_args(argv)
    args = ["--to-db"] + (["--model", a.model] if a.model else [])

    def job():
        try:
            batch.main(args)
        finally:                      # the comparison runs even if the forecast step failed
            actuals.main(["--to-db"])

    run_forever(job, minute=a.minute)


if __name__ == "__main__":
    main()
