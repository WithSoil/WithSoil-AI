import json
import os
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from schemas.recommendation import CropReasonList

class RecommendService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        self.structured_llm = self.llm.with_structured_output(CropReasonList)
        
        base_dir = Path(__file__).resolve().parent.parent
        json_file_path = base_dir / "data" / "recommendation_api_response_v1.json"
        self.crop_data_db = {}
        if json_file_path.exists():
            with open(json_file_path, "r", encoding="utf-8") as f:
                self.crop_data_db = json.load(f)

    def get_recommendation(self, region: str, purpose: str) -> list:
        if region not in self.crop_data_db:
            raise ValueError(f"'{region}' 지역의 추천 데이터가 없습니다.")

        raw_context = self.crop_data_db[region]
        context_str = json.dumps(raw_context, ensure_ascii=False)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 충청북도 농업기술원 소속의 베테랑 AI 농업 컨설턴트입니다. 제공된 [공공데이터 기반 추천 작물 정보]의 범위 내에서만 사용자의 재배 목적을 분석하여 추천 사유를 작성하세요."),
            ("human", "[공공데이터]\n{context}\n\n사용자 지역: {region}\n재배 목적 및 상황: {purpose}")
        ])

        chain = prompt | self.structured_llm
        result: CropReasonList = chain.invoke({
            "context": context_str,
            "region": region,
            "purpose": purpose
        })

        final_recommended_crops = []
        for ai_item in result.reasons:
            static_info = next((item for item in raw_context if item["crop_name"] == ai_item.crop_name), {})
            
            final_recommended_crops.append({
                "cropName": ai_item.crop_name,
                "recommendScore": ai_item.recommend_score or str(static_info.get("recommend_score", 0)),
                "aiReasonTitle": ai_item.ai_reason_title,
                "aiReasonDetail": ai_item.ai_reason_detail or static_info.get("guide_summary", ""),
                "difficultyLevel": static_info.get("difficulty_level", "-"),
                "optimalTemp": static_info.get("optimal_temp", "-"),
                "soilPh": static_info.get("soil_ph", "-"),
                "cultivationPeriod": static_info.get("cultivation_period", "-"),
                "mainTasks": static_info.get("main_tasks", []),
                "mainRisks": static_info.get("main_risks", [])
            })
            
        return final_recommended_crops