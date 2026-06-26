import os
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types 
from dotenv import load_dotenv
import traceback

load_dotenv()

router = APIRouter(
    prefix="/api/v1/ai",
    tags=["Recommendation"]
)

class RecommendRequest(BaseModel):
    region: str
    purpose: str

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = Path(__file__).resolve().parent.parent 
JSON_FILE_PATH = BASE_DIR / "recommendation_api_response_v1.json"
CROP_DATA_DB = {}

if JSON_FILE_PATH.exists():
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        CROP_DATA_DB = json.load(f)
    print(f"✅ 작물 추천 JSON 데이터 로드 완료 (총 {len(CROP_DATA_DB)}개 지역)")
else:
    print(f"⚠️ 경고: {JSON_FILE_PATH} 파일을 찾을 수 없습니다.")

@router.post("/recommend")
async def recommend_crop(request: RecommendRequest):
    try:
        if request.region not in CROP_DATA_DB:
            raise HTTPException(
                status_code=404, 
                detail=f"'{request.region}' 지역의 추천 데이터가 없습니다."
            )
        
        raw_context = CROP_DATA_DB[request.region]
        context_str = json.dumps(raw_context, ensure_ascii=False, indent=2)

        prompt = f"""당신은 충청북도 농업기술원 소속의 베테랑 AI 농업 컨설턴트입니다.
제공된 [공공데이터 기반 추천 작물 정보]의 작물(TOP3) 범위 내에서만 사용자의 질문에 답하세요.

[공공데이터 기반 추천 작물 정보]
{context_str}

사용자 지역: {request.region}
사용자 재배 목적 및 상황: {request.purpose}

위 작물들을 사용자의 목적에 맞게 추천하는 이유를 작성해주세요.
반드시 아래와 같은 JSON 배열 형식으로만 응답해야 하며, ```json 등의 마크다운 기호나 추가 설명은 절대 넣지 마세요.

[
  {{
    "crop_name": "작물이름(예: 고추)",
    "recommend_score": "95",
    "ai_reason_title": "왜 나한테 맞을까요?",
    "ai_reason_detail": "사용자의 목적에 맞춘 상세하고 친절한 추천 사유..."
  }}
]
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}]
            )
        )

        llm_response_text = response.text.replace("```json", "").replace("```", "").strip()
        ai_parsed_data = json.loads(llm_response_text)

        final_recommended_crops = []
        for ai_item in ai_parsed_data:
            crop_name = ai_item.get("crop_name")
            static_info = next((item for item in raw_context if item["crop_name"] == crop_name), {})
            
            merged_crop = {
                "cropName": crop_name,
                "recommendScore": ai_item.get("recommend_score", str(static_info.get("recommend_score", 0))),
                "aiReasonTitle": ai_item.get("ai_reason_title", "추천 사유"),
                "aiReasonDetail": ai_item.get("ai_reason_detail", static_info.get("guide_summary", "")),
                "difficultyLevel": static_info.get("difficulty_level", "-"),
                "optimalTemp": static_info.get("optimal_temp", "-"),
                "soilPh": static_info.get("soil_ph", "-"),
                "cultivationPeriod": static_info.get("cultivation_period", "-"),
                "mainTasks": static_info.get("main_tasks", []),
                "mainRisks": static_info.get("main_risks", [])
            }
            final_recommended_crops.append(merged_crop)

        return {
            "region": request.region,
            "purpose": request.purpose,
            "recommendedCrops": final_recommended_crops
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"작물 추천 중 오류 발생: {str(e)}")