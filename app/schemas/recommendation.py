from pydantic import BaseModel, Field
from typing import List

class RecommendRequest(BaseModel):
    region: str
    purpose: str

# LLM에게 강제할 출력 스키마 정의
class CropReason(BaseModel):
    crop_name: str = Field(description="작물이름(예: 고추)")
    recommend_score: str = Field(description="추천 점수 (0~100)")
    ai_reason_title: str = Field(description="왜 나한테 맞을까요?")
    ai_reason_detail: str = Field(description="사용자의 목적에 맞춘 상세하고 친절한 추천 사유")

class CropReasonList(BaseModel):
    reasons: List[CropReason]