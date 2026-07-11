import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/jobs", tags=["jobs"])

DB_PATH = Path(__file__).resolve().parent / "jarvis_jobs.db"
BRIDGE_TOKEN = os.getenv("JARVIS_BRIDGE_TOKEN", "")

class CreateJobRequest(BaseModel):
    command: str = Field(min_length=1, max_length=20000)

class JobUpdateRequest(BaseModel):
    status: str
    message: str = ""
    result: str = ""

def _auth(x_jarvis_token: str | None) -> None:
    if not BRIDGE_TOKEN:
        raise HTTPException(status_code=500, detail="JARVIS_BRIDGE_TOKEN is not configured.")
    if x_jarvis_token != BRIDGE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid Jarvis bridge token.")

def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            command TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT NOT NULL DEFAULT '',
            result TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    return conn

@router.post("")
def create_job(request: CreateJobRequest, x_jarvis_token: str | None = Header(default=None)):
    _auth(x_jarvis_token)
    now = datetime.now(timezone.utc).isoformat()
    job_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, command, status, created_at, updated_at) VALUES (?, ?, 'queued', ?, ?)",
            (job_id, request.command, now, now),
        )
    return {"id": job_id, "status": "queued"}

@router.get("/next")
def next_job(x_jarvis_token: str | None = Header(default=None)):
    _auth(x_jarvis_token)
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return {"job": None}
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE jobs SET status='claimed', updated_at=? WHERE id=?",
            (now, row["id"]),
        )
        data = dict(row)
        data["status"] = "claimed"
        return {"job": data}

@router.patch("/{job_id}")
def update_job(job_id: str, request: JobUpdateRequest, x_jarvis_token: str | None = Header(default=None)):
    _auth(x_jarvis_token)
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        found = conn.execute("SELECT id FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not found:
            raise HTTPException(status_code=404, detail="Job not found.")
        conn.execute(
            "UPDATE jobs SET status=?, message=?, result=?, updated_at=? WHERE id=?",
            (request.status, request.message, request.result, now, job_id),
        )
    return {"id": job_id, "status": request.status}

@router.get("/{job_id}")
def get_job(job_id: str, x_jarvis_token: str | None = Header(default=None)):
    _auth(x_jarvis_token)
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found.")
        return dict(row)
