"""Create the database tables, and optionally copy an existing forecasts.csv into them.

Run from the project root (the database must be running: `docker compose up -d db`):
    python -m hawa.db.init
    python -m hawa.db.init --from-csv data/forecasts/forecasts.csv     # one-time backfill of the old CSV
"""
import argparse

from hawa import config
from hawa.db import store


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-csv", help="forecasts.csv to copy into the database")
    p.add_argument("--url", default=config.DATABASE_URL, help="default: DATABASE_URL from .env / environment")
    a = p.parse_args(argv)

    engine = store.make_engine(a.url)
    store.init_db(engine)
    print(f"tables ready in {engine.url.render_as_string(hide_password=True)}")
    if a.from_csv:
        print(f"copied {store.load_csv(engine, a.from_csv)} rows from {a.from_csv}")


if __name__ == "__main__":
    main()
