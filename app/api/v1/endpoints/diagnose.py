# api/v1/endpoints/diagnose.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from services.disease_service import DiseaseService
import traceback

router = APIRouter()
disease_service = DiseaseService()

@router.get("/crops")
def get_supported_crops():
    crops = disease_service.get_supported_crops()
    if crops is None:
        raise HTTPException(status_code=503, detail="진단 모델이 아직 준비되지 않았습니다.")
    return {"crops": crops}

@router.post("/diagnose")
async def diagnose_crops(
    crop_name: str = Form(...),
    file: UploadFile = File(...),
    topk: int = Form(5)
):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

        image_bytes = await file.read()
        result = disease_service.predict_disease(
            image_bytes=image_bytes,
            crop_name=crop_name,
            topk=topk
        )

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