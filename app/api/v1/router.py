from fastapi import APIRouter
from api.v1.endpoints import recommend, diagnose, chat

api_router = APIRouter()

api_router.include_router(recommend.router, prefix="/ai", tags=["Recommendation"])
api_router.include_router(diagnose.router, prefix="/ai", tags=["Diagnosis"])
api_router.include_router(chat.router, prefix="/rag", tags=["Chat & RAG"])