"""Shared test fixtures.

`db_engine`: a fresh database for one test. SQLite file by default. To run the same tests on a real PostgreSQL,
set HAWA_TEST_DATABASE_URL to an EMPTY test database (its tables are dropped and recreated before each test):
    HAWA_TEST_DATABASE_URL=postgresql+psycopg://hawa:hawa@localhost:5433/hawa_test pytest tests/test_db_api.py
"""
import os

import pytest
import sqlalchemy as sa

from hawa.db import schema, store


@pytest.fixture
def db_engine(tmp_path):
    url = os.environ.get("HAWA_TEST_DATABASE_URL")
    engine = sa.create_engine(url) if url else sa.create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    schema.metadata.drop_all(engine)
    store.init_db(engine)
    yield engine
    engine.dispose()
