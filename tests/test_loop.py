import pandas as pd
import pytest

from hawa.forecasting import loop


def test_seconds_until_next_run():
    assert loop.seconds_until_next(pd.Timestamp("2026-11-18T06:00:00Z")) == 600           # exactly on the hour
    assert loop.seconds_until_next(pd.Timestamp("2026-11-18T06:04:30Z")) == 330
    assert loop.seconds_until_next(pd.Timestamp("2026-11-18T06:10:00Z")) == 3600          # just ran: next hour
    assert loop.seconds_until_next(pd.Timestamp("2026-11-18T06:55:00Z")) == 900           # crosses the hour
    assert loop.seconds_until_next(pd.Timestamp("2026-11-18T06:30:00Z"), minute=45) == 900


def test_loop_runs_now_then_sleeps_until_next_slot():
    times = iter(pd.to_datetime(["2026-11-18T06:44:00Z", "2026-11-18T06:44:20Z", "2026-11-18T07:10:05Z",
                                 "2026-11-18T07:10:30Z"]))
    sleeps, calls = [], []
    loop.run_forever(lambda: calls.append(1), clock=lambda: next(times), sleep=sleeps.append, max_runs=2)
    assert len(calls) == 2 and sleeps == [pytest.approx(25 * 60 + 40)]


def test_a_failing_run_does_not_stop_the_loop(capsys):
    calls = []

    def job():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("network down")
        if len(calls) == 2:
            raise SystemExit("No model found")

    loop.run_forever(job, clock=lambda: pd.Timestamp("2026-11-18T06:20:00Z"), sleep=lambda s: None, max_runs=3)
    out = capsys.readouterr().out
    assert len(calls) == 3 and "run failed: RuntimeError: network down" in out and "No model found" in out
