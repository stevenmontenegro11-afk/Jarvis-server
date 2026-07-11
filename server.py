import os
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="Jarvis Server")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")
client = OpenAI()

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20000)

class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=50)

class ChatResponse(BaseModel):
    reply: str

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        response = client.responses.create(
            model=MODEL,
            instructions=(
                "You are Jarvis, Steven's helpful personal AI assistant. "
                "Be conversational, practical, and concise. "
                "Never claim an action was completed unless a connected tool actually completed it."
            ),
            input=[{"role": m.role, "content": m.content} for m in request.messages],
        )
        reply = response.output_text.strip()
        if not reply:
            raise HTTPException(status_code=502, detail="Empty AI response.")
        return ChatResponse(reply=reply)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
