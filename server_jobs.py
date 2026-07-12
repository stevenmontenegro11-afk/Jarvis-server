from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/jobs", tags=["jobs"])

BRIDGE_TOKEN = os.getenv("JARVIS_BRIDGE_TOKEN", "")
SQLITE_TIMEOUT_SECONDS = 30.0


class CreateJobRequest(BaseModel):
    command: str = Field(min_length=1, max_length=20000)


class JobUpdateRequest(BaseModel):
    status: str = Field(min_length=1, max_length=100)
    message: str = Field(default="", max_length=100000)
    result: str = Field(default="", max_length=1000000)


def _resolve_db_path() -> Path:
    configured_path = os.getenv("JARVIS_JOBS_DB_PATH", "").strip()
    if configured_path:
        return Path(configured_path).expanduser()

    configured_data_dir = os.getenv("JARVIS_DATA_DIR", "").strip()
    if configured_data_dir:
        return Path(configured_data_dir).expanduser() / "jarvis_jobs.db"

    if os.getenv("RENDER", "").lower() in {"1", "true", "yes"}:
        return Path("/var/data/jarvis_jobs.db")

    return Path(__file__).resolve().parent / "jarvis_jobs.db"


DB_PATH = _resolve_db_path()


def _auth(x_jarvis_token: str | None) -> None:
    if not BRIDGE_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="JARVIS_BRIDGE_TOKEN is not configured.",
        )
    if x_jarvis_token != BRIDGE_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid Jarvis bridge token.",
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            DB_PATH,
            timeout=SQLITE_TIMEOUT_SECONDS,
            isolation_level=None,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to open persistent job database at {DB_PATH}. "
            "On Render, attach a persistent disk at /var/data or set "
            "JARVIS_JOBS_DB_PATH to a path on the persistent disk."
        ) from exc

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _initialize_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                command TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT NOT NULL DEFAULT '',
                result TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_jobs_status_created_at
            ON jobs (status, created_at)
            """
        )


def _job_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "command": row["command"],
        "status": row["status"],
        "message": row["message"],
        "result": row["result"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


_initialize_database()


@router.post("")
def create_job(
    request: CreateJobRequest,
    x_jarvis_token: str | None = Header(default=None),
):
    _auth(x_jarvis_token)
    now = _utc_now()
    job_id = str(uuid.uuid4())

    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                INSERT INTO jobs (
                    id,
                    command,
                    status,
                    message,
                    result,
                    created_at,
                    updated_at
                ) VALUES (?, ?, 'queued', '', '', ?, ?)
                """,
                (job_id, request.command, now, now),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    return {"id": job_id, "status": "queued"}


@router.get("/next")
def next_job(x_jarvis_token: str | None = Header(default=None)):
    _auth(x_jarvis_token)

    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                """
                SELECT *
                FROM jobs
                WHERE status = 'queued'
                ORDER BY created_at ASC
                LIMIT 1
                """
            ).fetchone()

            if row is None:
                connection.commit()
                return {"job": None}

            now = _utc_now()
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = 'claimed', updated_at = ?
                WHERE id = ? AND status = 'queued'
                """,
                (now, row["id"]),
            )

            if updated.rowcount != 1:
                connection.rollback()
                return {"job": None}

            claimed_row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (row["id"],),
            ).fetchone()
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    if claimed_row is None:
        return {"job": None}
    return {"job": _job_dict(claimed_row)}


@router.patch("/{job_id}")
def update_job(
    job_id: str,
    request: JobUpdateRequest,
    x_jarvis_token: str | None = Header(default=None),
):
    _auth(x_jarvis_token)
    now = _utc_now()

    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            updated = connection.execute(
                """
                UPDATE jobs
                SET status = ?, message = ?, result = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    request.status,
                    request.message,
                    request.result,
                    now,
                    job_id,
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise HTTPException(status_code=404, detail="Job not found.")
            connection.commit()
        except HTTPException:
            raise
        except Exception:
            connection.rollback()
            raise

    return {"id": job_id, "status": request.status}


@router.get("/{job_id}")
def get_job(
    job_id: str,
    x_jarvis_token: str | None = Header(default=None),
):
    _auth(x_jarvis_token)

    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_dict(row)
