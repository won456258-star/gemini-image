import json
import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# --- [필수 라이브러리 임포트] ---
from fastapi import Response, File, UploadFile, Form, HTTPException, Query, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image 
from google import genai
from google.genai import types
from rembg import remove  # 배경 제거 라이브러리
from genai_image import nano_banana_style_image_editing  # 이미지 생성 함수

# (기존 프로젝트 모듈 임포트 - 파일이 없으면 주석 처리하세요)
from base_dir import BASE_PUBLIC_DIR
from classes import PromptDeviderProcessor, AnswerTemplateProcessor, ClientError, MakePromptTemplateProcessor, ModifyPromptTemplateProcessor, QuestionTemplateProcessor, SpecQuestionTemplateProcessor
from make_default_game_folder import create_project_structure
from make_dummy_image_asset import check_and_create_images_with_text
from make_dummy_sound_asset import copy_and_rename_sound_files
from save_chat import load_chat, save_chat
from snapshot_manager import create_version, find_current_version_from_file, restore_version
from tools.debug_print import debug_print
from tsc import check_typescript_compile_error

# FastAPI 앱 인스턴스 생성
app = FastAPI(title="Gemini Code Assistant API")

# ⚠️ CORS 설정
origins = [
    "http://localhost:3000",      # React 앱
    "http://localhost:8080",      # 게임 iframe
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# 환경 변수 로드
load_dotenv()

# [1] Gemini API 초기화 (채팅 및 이미지 분석용)
gemini_api_key = os.getenv('GEMINI_API_KEY')
# 기본 채팅용 모델
chat_model_name = "gemini-2.5-flash"

try:
    gemini_client = genai.Client(api_key=gemini_api_key)
except Exception as e:
    print(f"클라이언트 초기화 오류: {e}")
    print("환경 변수 GEMINI_API_KEY가 설정되었는지 확인해 주세요.")
    exit()

# 요청 모델 정의
class CodeRequest(BaseModel):
    message: str
    game_name: str

# 서버 상태 체크
@app.get("/")
async def root():
    return {"status": "healthy", "message": "Gemini Code Assistant API is running"}

# ... (기존 유틸리티 함수들은 생략, 필요시 기존 파일 내용 유지) ...
# ... (기존 API: /category, /process-code 등은 gemini_client를 그대로 사용) ...


# ================================================================================
#  [신규 기능] 이미지 생성(Azure DALL-E) 및 배경 제거(rembg) API
# ================================================================================

@app.post("/generate-image")
async def generate_image_api(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    """
    [이미지 생성 흐름]
    1. 프론트엔드에서 이미지와 프롬프트 수신
    2. Gemini (Vision)로 이미지 분석 (gemini-1.5-flash 모델 사용)
    3. genai_image.py 내부에서 Azure DALL-E 호출하여 이미지 생성
    """
    # 이미지 분석에 특화된 가벼운 모델 사용
    vision_model_name = "gemini-1.5-flash"

    try:
        # 1. 업로드된 이미지 읽기
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 2. 이미지 생성 로직 호출
        result_bytes = nano_banana_style_image_editing(
            gemini_client=gemini_client,   # 분석용 Gemini 클라이언트 전달
            model_name=vision_model_name,  # 분석용 모델명 전달
            reference_image=pil_image,
            editing_prompt=prompt
        )

        if result_bytes:
            # 3. 생성된 이미지를 PNG 포맷으로 반환
            return Response(content=result_bytes, media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail="이미지 생성에 실패했습니다.")

    except Exception as e:
        print(f"API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/remove-bg")
async def remove_background_api(image: UploadFile = File(...)):
    """
    [배경 제거]
    rembg 라이브러리를 사용하여 배경을 투명하게 만듭니다.
    """
    try:
        image_data = await image.read()
        
        # rembg로 배경 제거
        result_data = remove(image_data)
        
        return Response(content=result_data, media_type="image/png")
        
    except Exception as e:
        print(f"배경 제거 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 서버 실행
if __name__ == "__main__":
    import uvicorn
    print("서버를 시작합니다... http://localhost:8000")
    uvicorn.run(
        "gemini:app",
        host="0.0.0.0",
        port=8000,
        reload=True,      # 코드 변경 시 자동 재시작
        log_level="info", # 로그 레벨 설정
        workers=1
    )