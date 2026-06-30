from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    mode: str = "general"

class ChatResponse(BaseModel):
    status: str
    answer: str