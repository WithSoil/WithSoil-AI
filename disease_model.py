import os
from io import BytesIO
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CropDiseaseModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.transform = None
        self.idx_to_class = None
        self.crop_to_indices = None

    def load(self):
        hf_model_id = os.getenv("HF_MODEL_ID")
        hf_model_file = os.getenv(
            "HF_MODEL_FILE",
            "best_efficientnet_b3_with_normal_20crops.pth"
        )

        # HF_TOKEN이 비어 있으면 None 처리
        # public repo면 token 없이 다운로드 가능
        hf_token = os.getenv("HF_TOKEN")

        if hf_token is not None:
            hf_token = hf_token.strip()

        if not hf_token:
            hf_token = None

        if not hf_model_id:
            raise RuntimeError("HF_MODEL_ID 환경변수가 설정되지 않았습니다.")

        checkpoint_path = hf_hub_download(
            repo_id=hf_model_id,
            filename=hf_model_file,
            token=hf_token
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        num_classes = checkpoint["num_classes"]
        self.idx_to_class = checkpoint["idx_to_class"]
        img_size = checkpoint.get("img_size", 300)

        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model

        self.crop_to_indices = defaultdict(list)

        for idx, cls_name in self.idx_to_class.items():
            idx = int(idx)
            crop = cls_name.split("_")[0]
            self.crop_to_indices[crop].append(idx)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        print("작물 병해 모델 로드 완료")
        print("device:", self.device)
        print("num_classes:", num_classes)
        print("지원 작물:", sorted(self.crop_to_indices.keys()))

    def predict(self, image_bytes: bytes, crop_name: str, topk: int = 5):
        if self.model is None:
            raise RuntimeError("모델이 로드되지 않았습니다.")

        if crop_name not in self.crop_to_indices:
            raise ValueError(
                f"지원하지 않는 작물명입니다: {crop_name}. "
                f"지원 작물: {sorted(self.crop_to_indices.keys())}"
            )

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        candidate_indices = self.crop_to_indices[crop_name]
        candidate_probs = probs[candidate_indices]

        filtered_probs = candidate_probs / candidate_probs.sum()
        sorted_indices = torch.argsort(filtered_probs, descending=True)

        results = []

        for rank in range(min(topk, len(sorted_indices))):
            local_idx = sorted_indices[rank].item()
            global_idx = candidate_indices[local_idx]

            class_name = self.idx_to_class[global_idx]
            disease_name = class_name.split("_", 1)[1]

            results.append({
                "rank": rank + 1,
                "class_name": class_name,
                "crop": crop_name,
                "disease": disease_name,
                "crop_filtered_confidence": round(float(filtered_probs[local_idx].item()), 6),
                "global_confidence": round(float(probs[global_idx].item()), 6),
            })

        return results

    def make_result(self, top_result: dict) -> dict:
        disease = top_result["disease"]
        confidence = top_result["crop_filtered_confidence"]
        crop = top_result["crop"]

        if confidence >= 0.8:
            if disease == "정상":
                return {
                    "result_type": "normal",
                    "diagnosis": "정상",
                    "message": f"{crop}은(는) 정상으로 보입니다.",
                    "confidence": confidence,
                }
            return {
                "result_type": "disease",
                "diagnosis": disease,
                "message": f"{crop}에 {disease}이(가) 의심됩니다.",
                "confidence": confidence,
            }

        if confidence >= 0.6:
            if disease == "정상":
                return {
                    "result_type": "uncertain",
                    "diagnosis": "정상 (불확실)",
                    "message": "정상으로 보이지만 확실하지 않습니다. 추가 촬영을 권장합니다.",
                    "confidence": confidence,
                }
            return {
                "result_type": "uncertain",
                "diagnosis": disease,
                "message": f"{disease} 가능성이 있으나 확실하지 않습니다. 추가 촬영을 권장합니다.",
                "confidence": confidence,
            }

        return {
            "result_type": "low_confidence",
            "diagnosis": None,
            "message": "사진이 선명하지 않아 진단이 어렵습니다. 병반이 잘 보이도록 가까이서 다시 촬영해 주세요.",
            "confidence": confidence,
        }