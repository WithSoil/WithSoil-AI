# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from google import genai
from google.genai import types 
from dotenv import load_dotenv
import traceback
from google.genai.errors import APIError
from disease_model import CropDiseaseModel

load_dotenv()

app = FastAPI(title="SmartFarm AI Server")

class ChatRequest(BaseModel):
    query: str
    mode: str = "general"  

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# 작물 병해 진단 모델 객체 생성
crop_disease_model = CropDiseaseModel()

@app.on_event("startup")
def startup_event():
    crop_disease_model.load()

@app.get("/")
def read_root():
    return {"message": "FastAPI AI Server is running"}

@app.get("/api/v1/ai/crops")
def get_supported_crops():
    if crop_disease_model.crop_to_indices is None:
        raise HTTPException(status_code=503, detail="진단 모델이 아직 준비되지 않았습니다.")
    return {"crops": sorted(crop_disease_model.crop_to_indices.keys())}

# 이곳에 작물 병해 진단 모델 업로드해주시면 될것같습니다.
@app.post("/api/v1/ai/diagnose")
async def diagnose_crops(
    crop_name: str = Form(...),
    file: UploadFile = File(...),
    topk: int = Form(5)
):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

        image_bytes = await file.read()

        predictions = crop_disease_model.predict(
            image_bytes=image_bytes,
            crop_name=crop_name,
            topk=topk
        )

        result = crop_disease_model.make_result(predictions[0])

        return {
            "status": "success",
            "crop": crop_name,
            "result_type": result["result_type"],
            "diagnosis": result["diagnosis"],
            "message": result["message"],
            "confidence": result["confidence"],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"진단 중 오류 발생: {str(e)}")

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
