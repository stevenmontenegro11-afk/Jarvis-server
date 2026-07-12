import os
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
from server_jobs import router as jobs_router
app.include_router(image_search_router)
app = FastAPI(title="Jarvis Server", version="1.1.0")
app.include_router(jobs_router)
app.include_router(image_search_router)

TEXT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")
client = OpenAI()


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=100)


class ChatResponse(BaseModel):
    reply: str


def needs_live_search(text: str) -> bool:
    text = text.lower()
    live_terms = (
        "today", "yesterday", "tomorrow", "latest", "current", "right now",
        "live", "score", "won", "winner", "game", "match", "schedule",
        "standings", "news", "weather", "price", "ufc", "soccer",
        "football", "nba", "nfl", "mlb", "nhl", "world cup"
    )
    return any(term in text for term in live_terms)


@app.get("/health")
def health():
    phoenix_now = datetime.now(ZoneInfo("America/Phoenix"))
    return {
        "status": "ok",
        "model": TEXT_MODEL,
        "phoenix_time": phoenix_now.isoformat(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        latest_message = request.messages[-1].content
        phoenix_now = datetime.now(ZoneInfo("America/Phoenix"))
       tools = [{"type": "web_search"}]
        response = client.responses.create(
            model=TEXT_MODEL,
            instructions=(
    "You are Jarvis, a helpful multi-user personal AI assistant. "
    "Be direct, conversational, practical, and accurate. "
    "Use the current conversation to understand what the user means. "
    "Do not hard-code or assume any user's name, interests, location, or preferences. "
    "Personalization must come only from the current user's conversation, account profile, permissions, and memory. "
    "Use web search whenever external, current, local, visual, uncertain, or rapidly changing information may help. "
    "Use web search for news, sports, schedules, weather, prices, products, businesses, public figures, locations, images, and recent events. "
    "For image requests, search the exact requested subject, verify that the source identifies the subject, and never invent a caption or show an unrelated image. "
    "Never claim an action was completed unless a connected tool actually completed and verified it. "
),
            input=[
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            tools=tools,
        )

        reply = response.output_text.strip()
        if not reply:
            raise HTTPException(status_code=502, detail="Empty AI response.")

        return ChatResponse(reply=reply)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
