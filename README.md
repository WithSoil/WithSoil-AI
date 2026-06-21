## 로컬 개발 환경 세팅 가이드

저장소를 클론(`git clone`)한 후, 로컬 컴퓨터에서 서버를 구동하기 위해 아래 순서대로 세팅을 진행해 주세요. (가상환경 `venv` 폴더는 깃허브 제외 대상이므로 각자 생성해야 합니다.)

### 1. 파이썬 가상환경(venv) 생성 및 활성화

**Windows (VS Code 터미널 또는 CMD):**
```bash
# 1. venv 가상환경 생성 (최초 1회만)
python -m venv venv

# 2. 가상환경 활성화
.\venv\Scripts\activate
```

**Mac / Linux:**
```bash
# 1. venv 가상환경 생성 (최초 1회만)
python3 -m venv venv

# 2. 가상환경 활성화
source venv/bin/activate
```
*💡 가상환경이 정상적으로 켜지면 터미널 창 맨 앞에 `(venv)` 문구가 뜹니다.*

### 2. 필수 라이브러리 일괄 설치
가상환경이 활성화된 상태에서, 패키지 명세서를 통해 의존성을 한 방에 설치합니다.
```bash
pip install -r requirements.txt
```

### 3. 환경 변수 (.env) 설정
저장소에는 실제 키 값 대신 템플릿 파일 `.env.example`만 올라가 있습니다. 이를 복사해서 `.env`를 만든 뒤 본인의 값으로 채워주세요. (`.env` 파일은 보안을 위해 절대 깃허브에 올리지 마세요! `.gitignore`에 등록되어 있습니다.)

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

`.env` 안에 채워야 할 값:
```text
# Gemini API 키 (각자 발급)
GEMINI_API_KEY=your_actual_gemini_api_key_here

# 작물 병해 진단 모델이 올라가 있는 Hugging Face repo 정보
HF_MODEL_ID=your_huggingface_repo_id_here
HF_MODEL_FILE=best_efficientnet_b3_with_normal_20crops.pth

# HF repo가 private일 때만 필요 (public이면 비워둬도 됨)
HF_TOKEN=
```
- `GEMINI_API_KEY`는 [Google AI Studio](https://aistudio.google.com/)에서 각자 발급받아 사용하세요.
- `HF_MODEL_ID` / `HF_MODEL_FILE` / `HF_TOKEN` 실제 값은 보안상 git이 아닌 팀 슬랙/노션 등으로 공유받아 입력하세요.

---

## 🏃‍♂️ 서버 구동 방법
가상환경이 켜진 상태에서 아래 명령어를 입력하면 로컬 서버가 가동됩니다.
```bash
uvicorn main:app --reload --port 8000
```
- 서버가 켜지면 브라우저에서 `http://127.0.0.1:8000/docs`로 접속하여 Swagger API 문서 인터페이스를 바로 확인할 수 있습니다.

---

## 🗺️ API 엔드포인트 명세 (1차 완공)

| Method | Endpoint | Description | Status |
| :--- | :--- | :--- | :--- |
| **GET** | `/` | 서버 구동 상태 확인용 헬스체크 | ✅ 완료 |
| **POST** | `/api/v1/ai/predict` | EfficientNet/YOLO 병해충 이미지 진단 및 AI 분석 | 🔨 조원 가중치 작업 중 |
| **POST** | `/api/v1/rag/chat` | 공공데이터 가이드 기반 RAG 및 일반 대화 에이전트 | ✅ 에러 처리 완비 |

---