"""One shared HTTP helper with retries, used by every ingestion module."""
import time

import requests


class ApiError(RuntimeError):
    """The API answered, but with an error we should not retry."""


def get_json(url, *, params=None, headers=None, timeout=90, retries=5,
             pause_s=1.0, rate_limit_wait_s=60):
    """GET a URL and return parsed JSON.

    Retries on: timeouts, connection errors, HTTP 429 (rate limit) and 5xx.
    Raises ApiError for other errors (bad request, wrong key, ...), with the
    reason the API gave. `pause_s` is a short sleep after every good call so
    we stay under the provider's rate limit.
    """
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            wait = min(10 * attempt, 60)
            print(f"  network problem ({type(e).__name__}); retry {attempt}/{retries} in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 429:
            print(f"  rate limited; waiting {rate_limit_wait_s}s")
            time.sleep(rate_limit_wait_s)
            continue
        if r.status_code >= 500:
            wait = 5 * attempt
            print(f"  server error {r.status_code}; retry {attempt}/{retries} in {wait}s")
            time.sleep(wait)
            continue
        time.sleep(pause_s)
        try:
            data = r.json()
        except ValueError:
            raise ApiError(f"HTTP {r.status_code}: response is not JSON: {r.text[:200]}")
        failed = r.status_code != 200 or (isinstance(data, dict) and data.get("error") is True)
        if failed:
            reason = data.get("reason") or data.get("message") or data if isinstance(data, dict) else data
            raise ApiError(f"HTTP {r.status_code}: {reason}")
        return data
    raise ApiError(f"gave up after {retries} attempts: {url}")
