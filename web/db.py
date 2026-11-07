"""Prediction history stored in PostgreSQL."""
import time

import psycopg
from psycopg.rows import dict_row

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            SERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    filename      TEXT        NOT NULL,
    label         TEXT        NOT NULL CHECK (label IN ('AI-generated', 'Real')),
    confidence    REAL        NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    prob_ai       REAL        NOT NULL,
    threshold     REAL        NOT NULL,
    model_name    TEXT        NOT NULL,
    model_version TEXT        NOT NULL,
    source        TEXT        NOT NULL,
    latency_ms    REAL        NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at DESC);
"""

INSERT = """
INSERT INTO predictions (filename, label, confidence, prob_ai, threshold, model_name, model_version, source, latency_ms)
VALUES (%(filename)s, %(label)s, %(confidence)s, %(prob_ai)s, %(threshold)s, %(model_name)s,
        %(model_version)s, %(source)s, %(latency_ms)s)
RETURNING *
"""
FIELDS = ('filename', 'label', 'confidence', 'prob_ai', 'threshold', 'model_name', 'model_version', 'source', 'latency_ms')


class PostgresStore:
    def __init__(self, url):
        self.url = url

    def _connect(self):
        return psycopg.connect(self.url, row_factory=dict_row, connect_timeout=5)

    def init(self, retries=10, delay=2):
        """Create the table, retrying while the database starts up."""
        for attempt in range(1, retries + 1):
            try:
                with self._connect() as conn:
                    conn.execute(SCHEMA)
                print('[db] schema ready', flush=True)
                return
            except psycopg.OperationalError as exc:
                print(f'[db] not ready ({attempt}/{retries}): {exc}', flush=True)
                time.sleep(delay)
        raise RuntimeError('could not connect to the database - check DATABASE_URL and the db container')

    def log(self, **result):
        with self._connect() as conn:
            return conn.execute(INSERT, {k: result[k] for k in FIELDS}).fetchone()

    def get(self, prediction_id):
        with self._connect() as conn:
            return conn.execute('SELECT * FROM predictions WHERE id = %s', (prediction_id,)).fetchone()

    def recent(self, limit=100):
        with self._connect() as conn:
            return conn.execute('SELECT * FROM predictions ORDER BY created_at DESC LIMIT %s', (limit,)).fetchall()

    def ping(self):
        try:
            with self._connect() as conn:
                conn.execute('SELECT 1')
            return True
        except psycopg.Error:
            return False