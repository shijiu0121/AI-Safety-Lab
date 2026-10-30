import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AEGIS_DATA_DIR", ROOT / "data")).resolve()
DATA.mkdir(parents=True, exist_ok=True)
DB = DATA / "aegis.sqlite3"
LOCK = threading.RLock()
TABLES = (
    "models",
    "experiments",
    "datasets",
    "batch_runs",
    "defenses",
    "calibrations",
    "defense_runs",
    "training_runs",
    "guard_sessions",
    "guard_events",
)


@contextmanager
def connection():
    """连接上下文需要主动 close；sqlite3 自带的上下文仅负责事务。"""
    with LOCK:
        database = sqlite3.connect(DB)
        try:
            with database:
                yield database
        finally:
            database.close()


def initialize():
    with connection() as c:
        for table in TABLES:
            c.execute(
                f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, body TEXT NOT NULL)"
            )
        for table in (
            "experiments",
            "batch_runs",
            "calibrations",
            "defense_runs",
            "training_runs",
        ):
            for key, raw in c.execute(f"SELECT id, body FROM {table}").fetchall():
                body = json.loads(raw)
                if body["status"] in ("queued", "running", "cancelling"):
                    body.update(
                        status="interrupted",
                        error="服务已重启。单图任务需重建；批量任务可检查恢复。",
                    )
                    c.execute(
                        f"UPDATE {table} SET body=? WHERE id=?", (json.dumps(body), key)
                    )

    for session in all_items("guard_sessions"):
        if session.get("active"):
            update(
                "guard_sessions",
                session["id"],
                active=False,
                stopped_reason="service restarted",
            )


def save(table, body):
    assert table in TABLES
    with connection() as c:
        c.execute(
            f"INSERT OR REPLACE INTO {table} (id,body) VALUES (?,?)",
            (body["id"], json.dumps(body, ensure_ascii=False)),
        )
    return body


def get(table, key):
    assert table in TABLES
    with connection() as c:
        row = c.execute(f"SELECT body FROM {table} WHERE id=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def all_items(table):
    assert table in TABLES
    with connection() as c:
        rows = c.execute(f"SELECT body FROM {table} ORDER BY rowid DESC").fetchall()
    return [json.loads(row[0]) for row in rows]


def update(table, key, **fields):
    with LOCK:
        item = get(table, key)
        if item is None:
            raise KeyError(key)
        item.update(fields)
        return save(table, item)
