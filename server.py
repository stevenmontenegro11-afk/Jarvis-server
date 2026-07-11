import os
import sqlite3
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "jarvis_memory.db"

TEXT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")

app = FastAPI(title="Jarvis Cloud Baseline", version="1.0.0")
client = OpenAI()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    conversation_id: str | None = None
    mode: Literal["auto", "text", "web", "image"] = "auto"


class ChatResponse(BaseModel):
    conversation_id: str
    type: Literal["text", "image"]
    reply: str | None = None
    image_base64: str | None = None
    mime_type: str | None = None


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )


def save_message(conversation_id: str, role: str, content: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )


def load_messages(conversation_id: str, limit: int = 40) -> list[dict[str, str]]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()

    return [
        {"role": role, "content": content}
        for role, content in reversed(rows)
    ]


def looks_like_image_request(text: str) -> bool:
    lowered = text.lower()
    phrases = (
        "generate an image",
        "create an image",
        "make a picture",
        "draw ",
        "show me a picture",
        "visualize ",
        "render ",
    )
    return any(phrase in lowered for phrase in phrases)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "text_model": TEXT_MODEL,
        "image_model": IMAGE_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid.uuid4())
    latest = request.messages[-1].content

    for message in request.messages:
        save_message(conversation_id, message.role, message.content)

    image_mode = request.mode == "image" or (
        request.mode == "auto" and looks_like_image_request(latest)
    )

    try:
        if image_mode:
            image = client.images.generate(
                model=IMAGE_MODEL,
                prompt=latest,
                size="1024x1024",
            )
            first = image.data[0]
            image_base64 = getattr(first, "b64_json", None)

            if not image_base64:
                raise RuntimeError("The image API did not return base64 image data.")

            save_message(conversation_id, "assistant", "[Generated image]")

            return ChatResponse(
                conversation_id=conversation_id,
                type="image",
                image_base64=image_base64,
                mime_type="image/png",
            )

        history = load_messages(conversation_id)
        tools = [{"type": "web_search"}] if request.mode in ("auto", "web") else []

        response = client.responses.create(
            model=TEXT_MODEL,
            instructions=(
                "You are Jarvis, Steven's personal AI assistant. "
                "Be practical and conversational. Use web search for current facts "
                "when it is available and needed. Never claim an external action "
                "was completed unless a connected tool actually completed it."
            ),
            input=history,
            tools=tools,
        )

        reply = response.output_text.strip()
        if not reply:
            raise RuntimeError("The AI returned an empty response.")

        save_message(conversation_id, "assistant", reply)

        return ChatResponse(
            conversation_id=conversation_id,
            type="text",
            reply=reply,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
