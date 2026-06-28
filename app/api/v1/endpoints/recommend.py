from fastapi import APIRouter, HTTPException
from schemas.recommendation import RecommendRequest
from services.recommend_service import RecommendService
import traceback

router = APIRouter()
recommend_service = RecommendService()

@router.post("/recommend")
async def recommend_crop(request: RecommendRequest):
    try:
        recommended_crops = recommend_service.get_recommendation(
            region=request.region, 
            purpose=request.purpose
        )
        
        return {
            "region": request.region,
            "purpose": request.purpose,
            "recommendedCrops": recommended_crops
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"작물 추천 중 오류 발생: {str(e)}")