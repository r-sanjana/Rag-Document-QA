import hashlib
import json
import pickle
import sqlite3
from pathlib import Path
from datetime import datetime

_LAST_STATUS = {
    "status": "NOT_CHECKED",
    "timestamp": None,
    "namespace": None,
}

BASE_DIR = Path(__file__).resolve().parent
CACHE_DB = BASE_DIR / ".rag_llm_cache.sqlite3"
STATUS_FILE = BASE_DIR / ".llm_cache_status.json"


def _connect():
    conn = sqlite3.connect(str(CACHE_DB), timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            value BLOB NOT NULL
        )
    """)
    return conn


def _make_key(namespace, key):
    payload = json.dumps(
        [str(namespace), key],
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _save_status(status, namespace=None):
    data = {
        "status": status,
        "namespace": str(namespace) if namespace is not None else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    _LAST_STATUS.update(data)  

    try:
        STATUS_FILE.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"LLM CACHE STATUS ERROR | {exc}")


def get(namespace, key):
    cache_key = _make_key(namespace, key)

    try:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT value
                FROM llm_cache
                WHERE cache_key = ? AND namespace = ?
                """,
                (cache_key, str(namespace)),
            ).fetchone()

        if row is None:
            _save_status("MISS", namespace)
            print(f"LLM CACHE MISS | {namespace}")
            return None

        value = pickle.loads(row[0])

        _save_status("HIT", namespace)
        print(f"LLM CACHE HIT | {namespace}")
        return value

    except (sqlite3.Error, pickle.UnpicklingError, EOFError, ValueError) as exc:
        _save_status("ERROR", namespace)
        print(f"LLM CACHE ERROR | {exc}")
        return None


def set(namespace, key, value):
    cache_key = _make_key(namespace, key)

    try:
        stored_value = pickle.dumps(value)

        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache
                    (cache_key, namespace, value)
                VALUES (?, ?, ?)
                """,
                (cache_key, str(namespace), stored_value),
            )
            conn.commit()

        _save_status("SAVED", namespace)
        print(f"LLM CACHE SAVED | {namespace}")

    except (sqlite3.Error, pickle.PickleError, TypeError, AttributeError) as exc:
        _save_status("ERROR", namespace)
        print(f"LLM CACHE SAVE ERROR | {exc}")


def get_last_cache_status():
    if not STATUS_FILE.exists():
        return {
            "status": "NOT_CHECKED",
            "namespace": None,
            "timestamp": None,
        }

    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {
            "status": "NOT_CHECKED",
            "namespace": None,
            "timestamp": None,
        }


def get_cache_stats():
    with _connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM llm_cache"
        ).fetchone()[0]

    return {
        "database": str(CACHE_DB),
        "cached_responses": count,
    }


def clear_cache():
    with _connect() as conn:
        conn.execute("DELETE FROM llm_cache")
        conn.commit()

    _save_status("CLEARED")
    print("LLM CACHE CLEARED")

def set_last_cache_status(status, namespace="gemini_text_v1"):
    _save_status(status, namespace)


def get_last_cache_status():
    try:
        if STATUS_FILE.exists():
            return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

    return _LAST_STATUS.copy()