# api/v1/endpoints/chat.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from schemas.chat import ChatRequest
from services.chat_service import ChatService
import traceback

router = APIRouter()
chat_service = ChatService()

@router.post("/chat")
async def chat_rag(request: ChatRequest):
    try:
        if request.mode == "general":
            answer = await chat_service.chat_general(request.query)
            return {"status": "success", "answer": answer}
            
        elif request.mode == "rag":
            answer = await chat_service.chat_rag(request.query)
            return {"status": "success", "answer": answer}
            
        else:
            raise HTTPException(status_code=400, detail="올바르지 않은 챗봇 모드입니다.")

    except Exception as e:
        traceback.print_exc()
        return {"status": "INTERNAL_SERVER_ERROR", "message": str(e)}

@router.post("/chat/image")
async def chat_rag_with_image(
    query: str = Form(...),
    file: UploadFile = File(...),
    mode: str = Form("general")
):
    try:
        if mode != "general":
            raise HTTPException(status_code=400, detail="이미지 질문은 일반 대화 모드만 지원합니다.")

        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

        image_bytes = await file.read()
        answer = await chat_service.chat_with_image(
            query=query,
            image_bytes=image_bytes,
            mime_type=file.content_type
        )

        return {"status": "success", "answer": answer}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return {"status": "INTERNAL_SERVER_ERROR", "message": str(e)}