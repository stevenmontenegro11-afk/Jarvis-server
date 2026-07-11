import os
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
from server_jobs import router as jobs_router

app = FastAPI(title="Jarvis Server", version="1.1.0")

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
        tools = [{"type": "web_search"}] if needs_live_search(latest_message) else []

        response = client.responses.create(
            model=TEXT_MODEL,
            instructions=(
                "You are Jarvis, Steven's personal AI assistant. "
                f"The current date and time in Steven's Arizona timezone is "
                f"{phoenix_now.strftime('%A, %B %d, %Y at %I:%M %p')}. "
                "Steven frequently asks about major soccer, UFC, NBA, NFL, and other sports events. "
                "For every question involving today, yesterday, current events, sports results, "
                "scores, schedules, weather, news, or prices, use web search before answering. "
                "Never rely on remembered dates or scores for current information. "
                "When a sports question is broad, such as 'who won the soccer game yesterday?', "
                "search the major or most prominent relevant event and answer with your best likely "
                "interpretation. Clearly say what match you assumed. Do not ask a clarifying question "
                "first unless web search cannot identify a reasonable likely event. "
                "Example style: 'Assuming you mean the World Cup quarterfinal, Spain beat Belgium 2-1.' "
                "Use the conversation history to infer what Steven is referring to. "
                "Be direct, conversational, and practical. "
                "Never claim an external action was completed unless a connected tool actually completed it."
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
