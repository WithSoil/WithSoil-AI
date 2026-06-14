# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types 
from dotenv import load_dotenv
import traceback
from google.genai.errors import APIError

load_dotenv()

app = FastAPI(title="SmartFarm AI Server")

class ChatRequest(BaseModel):
    query: str
    mode: str = "general"  

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")
def read_root():
    return {"message": "FastAPI AI Server is running"}

# 이곳에 작물 병해 진단 모델 업로드해주시면 될것같습니다.
@app.post("/api/v1/ai/diagnose")
async def diagnose_crops(file: UploadFile = File(...)):
    return {
        "status": "success",
        "filename": file.filename,
        "predictions": [
            {"label": "sample_object", "confidence": 0.88, "box": [50, 50, 150, 150]}
        ]
    }

@app.post("/api/v1/rag/chat")
async def chat_rag(request: ChatRequest):
    try:
        if request.mode == "general":
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                # model="gemini.3.5-flash",
                contents=request.query,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}]
                )
            )
            return {
                "status": "success",
                "answer": response.text  
            }
            
        elif request.mode == "rag":
            # TODO: 유사도 검색(Vector Search) 코드 추가.
            return {
                "status": "success",
                "answer": "농부 일지 및 공공데이터 RAG 엔진 뼈대입니다."
            }
        
        else:
            raise HTTPException(status_code=400, detail="올바르지 않은 챗봇 모드입니다.")

    except Exception as e:
            traceback.print_exc()
            
            error_status = "INTERNAL_SERVER_ERROR"
            
            if isinstance(e, APIError) and hasattr(e, 'message') and e.response_json:
                error_status = e.response_json.get('error', {}).get('status', 'GOOGLE_API_ERROR')
                
            return {
                "status": error_status,
                "message": str(e)
            }