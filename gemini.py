import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import io 
import os
import time
from dotenv import load_dotenv
from typing import List, Optional, Any 

# --- 추가된 라이브러리 ---
from fastapi import Response, File, UploadFile, Form, HTTPException, Query, Request, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from PIL import Image
from rembg import remove # 배경 제거 라이브러리
from genai_image import nano_banana_style_image_editing # 이미지 생성 함수
import ffmpeg

# 기존 모듈 임포트 유지
from base_dir import BASE_PUBLIC_DIR

# classes.py에서 필요한 도구들을 빠짐없이 가져옵니다.
from classes import (
    PromptDeviderProcessor, 
    AnswerTemplateProcessor, 
    ClientError, 
    MakePromptTemplateProcessor, 
    ModifyPromptTemplateProcessor, 
    QuestionTemplateProcessor, 
    SpecQuestionTemplateProcessor
)

from make_default_game_folder import create_project_structure
from make_dummy_image_asset import check_and_create_images_with_text
from make_dummy_sound_asset import copy_and_rename_sound_files
from save_chat import load_chat, save_chat
from snapshot_manager import create_version, find_current_version_from_file, restore_version
from tools.debug_print import debug_print
from tsc import check_typescript_compile_error

# 환경 변수 로드
load_dotenv()

# [Gemini 설정]
gemini_api_key = os.getenv('GEMINI_API_KEY')
model_name = "gemini-2.5-flash"

try:
    gemini_client = genai.Client(api_key=gemini_api_key)
except Exception as e:
    print(f"클라이언트 초기화 오류: {e}")
    exit()

# FastAPI 앱 인스턴스 생성
app = FastAPI(title="Gemini Code Assistant API")

# CORS 설정
origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# -------------------------------------------------------------------------
#  [데이터 모델 정의] - (사용하기 전에 먼저 정의되어야 함)
# -------------------------------------------------------------------------

class CodeRequest(BaseModel):
    message: str
    game_name: str

class RestoreRequest(BaseModel):
    version: str
    game_name: str

# 🌟 [추가됨] 누락되었던 RevertRequest 클래스 추가
class RevertRequest(BaseModel):
    game_name: str

class AssetItem(BaseModel):
    name: str
    url: str

class AssetsResponse(BaseModel):
    images: List[AssetItem]
    sounds: List[AssetItem]

class ErrorData(BaseModel):
    type: str
    message: str
    source: str
    lineno: int
    colno: int
    stack: str
    time: str
    game_version: str

class ErrorBatch(BaseModel):
    type: str
    game_name: str
    game_version: str
    collected_at: str
    error_count: int
    error_report: str 
    errors: List[ErrorData]

class DataUpdatePayload(BaseModel):
    game_name: str
    data: dict

class WrappedSubmitData(BaseModel):
    game_name: str
    payload: str

# -------------------------------------------------------------------------
#  [유틸리티 함수들]
# -------------------------------------------------------------------------

def remove_comments_from_file(file_path):
    if not os.path.exists(file_path): return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        code_string = f.read()
    code_string = re.sub(r'(?<![\'"])\#.*', '', code_string)
    code_string = re.sub(r'("""[\s\S]*?""")|(\'\'\'[\s\S]*?\'\'\')', '', code_string)
    code_string = re.sub(r'\n\s*\n', '\n', code_string).strip()
    return code_string

def remove_code_fences_safe(code_string: str) -> str:
    stripped_string = code_string.strip()
    content_start = 0
    if stripped_string.startswith('```'):
        stripped_string = stripped_string.replace('\\n', '\n')
        first_newline_index = stripped_string.find('\n')
        if first_newline_index != -1:
            content_start = first_newline_index + 1
        else:
            content_start = 3
    processed_string = stripped_string[content_start:]
    final_string = processed_string.rstrip()
    if final_string.endswith('```'):
        final_string = final_string[:-3]
    return final_string.strip()

def GAME_DIR(game_name:str): return BASE_PUBLIC_DIR / game_name
def CODE_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "game.ts"
def DATA_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "data.json"
def SPEC_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "spec.md"
def CHAT_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "chat.json"
def ASSETS_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "assets"
def ARCHIVE_LOG_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "archive" / "change_log.json"
CODE_PATH_NOCOMMENT = "" 

def parse_ai_code_response(response_text):
    result = {}
    code_start = response_text.find("###CODE_START###") + len("###CODE_START###")
    code_end = response_text.find("###CODE_END###")
    result['game_code'] = response_text[code_start:code_end].strip()
    data_start = response_text.find("###DATA_START###") + len("###DATA_START###")
    data_end = response_text.find("###DATA_END###")
    json_string = response_text[data_start:data_end].strip()
    result['game_data'] = json_string
    desc_start = response_text.find("###DESCRIPTION_START###") + len("###DESCRIPTION_START###")
    desc_end = response_text.find("###DESCRIPTION_END###")
    result['description'] = response_text[desc_start:desc_end].strip()
    return result

def parse_ai_qna_response(response_text):
    result = {}
    code_start = response_text.find("###COMMENT_START###") + len("###COMMENT_START###")
    code_end = response_text.find("###COMMENT_END###")
    result['comment'] = response_text[code_start:code_end].strip()
    code_start = response_text.find("###SPECIFICATION_START###") + len("###SPECIFICATION_START###")
    code_end = response_text.find("###SPECIFICATION_END###")
    result['specification'] = response_text[code_start:code_end].strip()
    return result

def parse_ai_answer_response(response_text):
    result = {}
    answer_start = response_text.find("###ANSWER_START###") + len("###ANSWER_START###")
    answer_end = response_text.find("###ANSWER_END###")
    result['answer'] = response_text[answer_start:answer_end].strip()
    return result

def validate_json(json_str):
    try:
        json.loads(json_str)
        return ""
    except json.JSONDecodeError as e:
        return f"{e.msg} (line {e.lineno}, col {e.colno})"

# (프로세서 초기화)
makePTP = MakePromptTemplateProcessor()
modifyPTP = ModifyPromptTemplateProcessor()
pdp = PromptDeviderProcessor()
qtp = QuestionTemplateProcessor()
sqtp = SpecQuestionTemplateProcessor()
atp = AnswerTemplateProcessor()

GAMES_ROOT_DIR = BASE_PUBLIC_DIR.resolve() 
STYLE_FILE_NAME = "style.txt" 

# [에셋 추론 함수]
async def find_best_matching_asset(message: str, game_name: str, gemini_client) -> tuple[str, str] | None:
    assets_dir = GAMES_ROOT_DIR / game_name / "assets"
    if not assets_dir.exists(): return None

    game_data_path = DATA_PATH(game_name)
    if not game_data_path.exists(): return None
    
    with open(game_data_path, 'r', encoding='utf-8') as f:
        game_data = json.load(f)

    image_assets = game_data.get('assets', {}).get('images', [])
    if not image_assets: return None

    for idx, asset in enumerate(image_assets):
        filename = os.path.basename(asset.get('path', ''))
        name = asset.get('name', '').lower()
        if filename in message or name in message:
            return str(idx), filename

    asset_list_str = "\n".join([f"- Index {i}: {a.get('name')} (File: {os.path.basename(a.get('path',''))})" for i, a in enumerate(image_assets)])
    
    prompt = f"""
    User Request: "{message}"
    Current Game Assets:
    {asset_list_str}
    Task: Identify which single asset the user wants to change.
    Return ONLY the Index number. If no asset matches, return -1.
    """
    
    try:
        print(f"   🧠 [Gemini 추론 중]...")
        response = gemini_client.models.generate_content(model=model_name, contents=prompt)
        result = response.text.strip()
        match = re.search(r'\d+', result)
        if match:
            idx = int(match.group())
            if 0 <= idx < len(image_assets):
                target_asset = image_assets[idx]
                fname = os.path.basename(target_asset.get('path', ''))
                print(f"   🎯 [추론 성공] 타겟 에셋: {fname} (Index: {idx})")
                return str(idx), fname
    except Exception as e:
        print(f"   ⚠️ 에셋 추론 실패: {e}")

    return None

# [에셋 재생성 로직]
async def _regenerate_asset_logic(game_name: str, asset_id: str, new_prompt: str):
    print(f"\n🎨 [AI 에셋 재생성 시작] 게임: {game_name}, 에셋 ID: {asset_id}")
    
    style_path = GAMES_ROOT_DIR / game_name / STYLE_FILE_NAME
    saved_style = ""
    if style_path.exists():
        with open(style_path, 'r', encoding='utf-8') as f: saved_style = f.read().strip()
            
    game_data_path = DATA_PATH(game_name)
    with open(game_data_path, 'r', encoding='utf-8') as f: game_data = json.load(f)
    
    images_to_process = game_data.get('assets', {}).get('images', [])
    if not images_to_process or int(asset_id) >= len(images_to_process):
        return False, f"❌ 오류: 에셋 ID '{asset_id}'를 찾을 수 없습니다."

    asset_info = images_to_process[int(asset_id)]
    asset_name = os.path.basename(asset_info.get('path', ''))
    current_image_path = GAMES_ROOT_DIR / game_name / "assets" / asset_name

    if not current_image_path.exists():
        return False, f"❌ 오류: '{asset_name}' 파일을 찾을 수 없습니다."

    final_prompt = new_prompt
    if saved_style:
        final_prompt = f"{new_prompt}. (Style: {saved_style})"
            
    print(f"   최종 AI 요청 프롬프트: {final_prompt}")

    try:
        ref_image = Image.open(current_image_path).convert("RGB")
        new_image_bytes = nano_banana_style_image_editing(
            gemini_client=gemini_client,
            model_name=model_name, 
            reference_image=ref_image,
            editing_prompt=final_prompt
        )

        if not new_image_bytes: return False, "❌ 이미지 생성 실패."

        if "background" not in asset_name.lower() and "bg" not in asset_name.lower():
            try:
                img_obj = Image.open(io.BytesIO(new_image_bytes)).convert("RGBA")
                removed = remove(img_obj)
                with io.BytesIO() as out:
                    removed.save(out, format="PNG")
                    new_image_bytes = out.getvalue()
            except: pass

        with open(current_image_path, "wb") as f: f.write(new_image_bytes)
        return True, f"✅ '{asset_name}' 변경 완료! ({new_prompt})"

    except Exception as e:
        return False, f"❌ 에러 발생: {str(e)}"

# [코드 수정 로직]
def modify_code(message, question, game_name):
    create_project_structure(GAME_DIR(game_name))
    original_code = ""
    if os.path.exists(CODE_PATH(game_name)):
        with open(CODE_PATH(game_name), 'r', encoding='utf-8') as f: original_code = f.read()
    original_data = ""
    if os.path.exists(DATA_PATH(game_name)):
        with open(DATA_PATH(game_name), 'r', encoding='utf-8') as f: original_data = f.read()

    request_obj = type('obj', (object,), {'message': message, 'game_name': game_name})
    prompt = makePTP.get_final_prompt(request_obj, question) if original_code == "" else modifyPTP.get_final_prompt(request_obj, question, original_code, original_data)

    print(f"AI 모델이 작업 중 입니다: {model_name}...")
    response = gemini_client.models.generate_content(model=model_name, contents=prompt)
    responseData = parse_ai_code_response(response.text)
    
    game_code = remove_code_fences_safe(responseData.get('game_code', ''))
    game_data_str = remove_code_fences_safe(responseData.get('game_data', ''))
    description = remove_code_fences_safe(responseData.get('description', ''))

    if game_code:
        os.makedirs(os.path.dirname(CODE_PATH(game_name)), exist_ok=True)
        with open(CODE_PATH(game_name), 'w', encoding='utf-8') as f: f.write(game_code)

    error = ""
    if game_data_str:    
        error = validate_json(game_data_str)
        json_data = {}
        if not error: json_data = json.loads(game_data_str)
        
        regen_keywords = ["전부", "모든", "다시", "새로", "초기화"]
        should_force_regen = any(k in message for k in regen_keywords)
        
        if should_force_regen:
            assets_path = GAME_DIR(game_name) / "assets"
            if assets_path.exists():
                try:
                    for item in os.listdir(assets_path):
                        path = os.path.join(assets_path, item)
                        if os.path.isdir(path): shutil.rmtree(path)
                        else: os.remove(path)
                except: pass

        check_and_create_images_with_text(
            json_data, 
            GAME_DIR(game_name), 
            theme_context=message, 
            is_force=should_force_regen,
            game_data_full=json_data,
            gemini_client=gemini_client,
            model_name=model_name
        )
        
        copy_and_rename_sound_files(json_data, GAME_DIR(game_name))
        os.makedirs(os.path.dirname(DATA_PATH(game_name)), exist_ok=True)
        with open(DATA_PATH(game_name), 'w', encoding='utf-8') as f: f.write(game_data_str)

    if not error: error = check_typescript_compile_error(CODE_PATH(game_name))
    return game_code, game_data_str, description, error

def describe_code(request: CodeRequest):
    code = remove_comments_from_file(CODE_PATH(request.game_name))
    if code == "": return "분석할 코드가 없습니다."
    prompt = request.message + """ 이 것은 아래의 코드에 대한 질문입니다.
    답변은 반드시 다음과 같은 json 형식으로 해주세요: {response:str}""" + "\n\n<TypeScript code>\n" + code
    response = gemini_client.models.generate_content(model=model_name, contents=prompt)
    reply_content = json.loads(remove_code_fences_safe(response.text))
    return reply_content['response']

@app.post("/category")
async def category(request: CodeRequest):
    prompt = f"[사용자쿼리: {request.message}]\n" + """
    이 앱은 사용자의 자연어 입력을 받아 게임을 만드는 앱입니다.
    카테고리 분류: 1:수정요청, 2:질문, 3:기타, 4:부적절
    응답형식: {"category": int, "dscription": str, "response": str}
    """
    response = gemini_client.models.generate_content(model=model_name, contents=prompt)
    return json.loads(remove_code_fences_safe(response.text))

# -------------------------------------------------------------------------
#  [API: 채팅 및 코드 수정]
# -------------------------------------------------------------------------
@app.post("/process-code")
async def process_code(request: CodeRequest):
    game_name = request.game_name
    message = request.message
    
    if message.startswith("스타일 설정:") or message.startswith("Set style:"):
        style_content = message.split(":", 1)[1].strip()
        style_path = GAMES_ROOT_DIR / game_name / STYLE_FILE_NAME
        if not style_path.parent.exists(): style_path.parent.mkdir(parents=True, exist_ok=True)
        with open(style_path, 'w', encoding='utf-8') as f: f.write(style_content)
        return {"status": "success", "reply": f"✅ 스타일 설정 완료: {style_content}"}

    # 스마트 에셋 변경 감지
    asset_match = re.search(r'([\w-]+\.png)', message)
    change_keywords = ["바꿔", "변경", "그려", "수정", "change"]
    is_change_request = any(k in message for k in change_keywords)

    if is_change_request:
        asset_id, asset_filename = None, None
        
        if asset_match: 
            filename = asset_match.group(1)
            game_data_path = DATA_PATH(game_name)
            if game_data_path.exists():
                with open(game_data_path, 'r', encoding='utf-8') as f: d = json.load(f)
                for i, a in enumerate(d.get('assets',{}).get('images',[])):
                    if os.path.basename(a.get('path','')) == filename:
                        asset_id = str(i); asset_filename = filename; break
        else: 
            matched = await find_best_matching_asset(message, game_name, gemini_client)
            if matched: asset_id, asset_filename = matched

        if asset_id:
            prompt = message.replace("바꿔줘", "").replace("변경해줘", "").strip()
            success, reply = await _regenerate_asset_logic(game_name, asset_id, prompt)
            save_chat(CHAT_PATH(game_name), "bot", reply)
            return {"status": "success" if success else "fail", "reply": reply}

    # 기본 코드 수정 로직
    prompt = pdp.get_final_prompt(request.message)
    success = False
    fail_message = ""
    for i in range(3):
        try:
            response = gemini_client.models.generate_content(model=model_name, contents=prompt)
            devide = json.loads(remove_code_fences_safe(response.text))
            reqs = devide.get("Modification_Requests", [])
            
            if reqs:
                user_req = "\n".join(reqs)
                code, data, desc, err = modify_code(user_req, "", game_name)
                save_chat(CHAT_PATH(game_name), "bot", desc)
                return {"status": "success", "code": code, "data": data, "reply": desc}
            else:
                return {"status": "success", "reply": "수정 요청이 없습니다."}
                
        except Exception as e:
            fail_message = str(e)
    
    return {"status": "fail", "reply": fail_message}

# -------------------------------------------------------------------------
#  [API: 에셋 및 데이터 조회 (404 해결용)]
# -------------------------------------------------------------------------
@app.get("/game_data")
async def get_game_data(game_name: str):
    if os.path.exists(DATA_PATH(game_name)):
         with open(DATA_PATH(game_name), 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

@app.get("/assets", response_model=AssetsResponse)
def get_assets(game_name: str = Query(..., alias="game_name")):
    assets_dir = GAMES_ROOT_DIR / game_name / "assets"
    images, sounds = [], []
    if assets_dir.is_dir():
        relative_url_base = f"/static/{game_name}/assets/" 
        for fn in os.listdir(assets_dir):
            file_path = assets_dir / fn
            if file_path.is_file():
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")):
                    images.append(AssetItem(name=fn, url=f"{relative_url_base}{fn}"))
                elif fn.lower().endswith((".mp3", ".wav", ".ogg", ".m4a", ".flac")):
                    sounds.append(AssetItem(name=fn, url=f"{relative_url_base}{fn}"))
    return AssetsResponse(images=images, sounds=sounds)

@app.get("/load-chat")
def load_chat_request(game_name: str = Query(..., min_length=1)):
    try: return load_chat(CHAT_PATH(game_name))
    except Exception: return {"chat": []}

@app.get("/spec")
async def get_spec(game_name: str):
    spec = " "
    if os.path.exists(SPEC_PATH(game_name)):
        with open(SPEC_PATH(game_name), 'r', encoding='utf-8') as f: spec = f.read()
    return spec

# -------------------------------------------------------------------------
#  [API: 기타 기능들]
# -------------------------------------------------------------------------
@app.post("/spec-question")
async def spec_question(request: CodeRequest):
    try:        
        old_spec = ""
        if os.path.exists(SPEC_PATH(request.game_name)):
            with open(SPEC_PATH(request.game_name), 'r', encoding='utf-8') as f: old_spec = f.read()
        prompt = sqtp.get_final_prompt("", request.message, old_spec)
        response = gemini_client.models.generate_content(model=model_name, contents=prompt)
        return {"reply": remove_code_fences_safe(response.text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/restore-version")
async def restore_version_request(request_data: RestoreRequest):    
    if not request_data.version: raise HTTPException(status_code=400, detail="버전 정보 누락")
    if restore_version(GAME_DIR(request_data.game_name), request_data.version):
        return JSONResponse(content={"status": "success", "message": "복원 성공"}, status_code=200)
    else:
        raise HTTPException(status_code=500, detail="복원 실패")

@app.get("/snapshot-log")
async def get_snapshot_log(game_name: str):    
    path = ARCHIVE_LOG_PATH(game_name)
    if not path.exists(): return {"versions":[]}
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

@app.post("/client-error")
async def receive_client_error(batch: ErrorBatch):
    print(batch.error_report)
    save_chat(CHAT_PATH(batch.game_name), "bot", batch.error_report)
    return {"status": "success"}

@app.post("/data-update")
async def data_update(update: DataUpdatePayload):
    with open(DATA_PATH(update.game_name), 'w', encoding='utf-8') as f:
        json.dump(update.data, f, ensure_ascii=False, indent=4)
    version_info = find_current_version_from_file(ARCHIVE_LOG_PATH(update.game_name))
    create_version(GAME_DIR(update.game_name), parent_name=version_info.get("version"), summary='게임 데이터 수정')
    return {"status": "success"}

@app.post("/qna")
async def qna_process(data: WrappedSubmitData):
    game_name = data.game_name
    chat_data = json.loads(data.payload)
    output_lines = []
    for i, item in enumerate(chat_data.get('mainQuestions', [])):
        output_lines.append(f"질문{i+1}: {item.get('question','')}\n답변{i+1}: {item.get('answer','미입력')}\n")
    for i, item in enumerate(chat_data.get('additionalRequests', [])):
        output_lines.append(f"추가요청{i+1}: {item.get('request','')}\n")
    result = "\n".join(output_lines)

    old_spec = ""
    if os.path.exists(SPEC_PATH(game_name)):
        with open(SPEC_PATH(game_name), 'r', encoding='utf-8') as f: old_spec = f.read()

    prompt = atp.get_final_prompt(old_spec, result)
    response = gemini_client.models.generate_content(model=model_name, contents=prompt)
    parse = parse_ai_qna_response(response.text)
    spec = parse['specification']
    
    if os.path.dirname(SPEC_PATH(game_name)): os.makedirs(os.path.dirname(SPEC_PATH(game_name)), exist_ok=True)
    with open(SPEC_PATH(game_name), 'w', encoding='utf-8') as f: f.write(spec)

    prompt = sqtp.get_final_prompt("", "", spec) 
    response = gemini_client.models.generate_content(model=model_name, contents=prompt)
    return {"status": "success", "reply": remove_code_fences_safe(response.text)}

@app.post("/revert")
async def revert_code(request: RevertRequest):
    version_info = find_current_version_from_file(ARCHIVE_LOG_PATH(request.game_name))
    restore_success = restore_version(GAME_DIR(request.game_name), version_info.get("parent"))
    if restore_success:
        save_chat(CHAT_PATH(request.game_name), "bot", "코드를 이전 버전으로 되돌렸습니다.")
        return {"status": "success", "reply": "되돌리기 성공"}
    else:
        return {"status": "success", "reply": "되돌릴 내역 없음"}

@app.post("/generate-image")
async def generate_image_api(prompt: str = Form(...), image: UploadFile = File(...)):
    vision_model_name = "gemini-2.5-flash" 
    try:
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        result_bytes = nano_banana_style_image_editing(
            gemini_client=gemini_client,
            model_name=vision_model_name,
            reference_image=pil_image,
            editing_prompt=prompt
        )
        if result_bytes: return Response(content=result_bytes, media_type="image/png")
        else: raise HTTPException(status_code=500, detail="이미지 생성 실패")
    except Exception as e:
        print(f"API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove-bg")
async def remove_background_api(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        result_data = remove(image_data)
        return Response(content=result_data, media_type="image/png")
    except Exception as e:
        print(f"배경 제거 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/static/{game_name}/{file_path:path}")
async def serve_selective_static_file(game_name: str, file_path: str):
    if not file_path.startswith("assets/"):
        raise HTTPException(status_code=403, detail="Access denied")
    full_path = GAMES_ROOT_DIR / game_name / file_path
    try:
        if not full_path.resolve().is_relative_to(GAMES_ROOT_DIR):
             raise HTTPException(status_code=403, detail="Invalid path")
    except Exception:
        raise HTTPException(status_code=404, detail="File Not Found")
    if full_path.is_file(): return FileResponse(full_path)
    else: raise HTTPException(status_code=404, detail="File Not Found")

def _is_safe_filename(name: str) -> bool:
    return name == os.path.basename(name) and not any(x in name for x in ["/", "\\"])

@app.post("/replace-asset")
async def replace_asset(
    game_name: str = Form(...),
    old_name: str = Form(...),
    type: str = Form(...),
    file: UploadFile = File(...),
):
    if not game_name.strip() or type not in ("image", "sound") or not _is_safe_filename(old_name):
        raise HTTPException(status_code=400, detail="Invalid request")
    assets_dir = (GAMES_ROOT_DIR / game_name / "assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    old_path = (assets_dir / old_name)
    new_name = f"{Path(old_name).stem}.{'png' if type == 'image' else 'mp3'}"
    dst_path = (assets_dir / new_name)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        ext = Path(file.filename).suffix.lower()
        if type == "image":
            if ext == ".png": shutil.copyfile(tmp_path, dst_path)
            else:
                with Image.open(tmp_path) as img:
                    img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB").save(dst_path, "PNG")
        else:
            if ext == ".mp3": shutil.copyfile(tmp_path, dst_path)
            else:
                subprocess.run(["ffmpeg", "-y", "-i", str(tmp_path), "-b:a", "192k", str(dst_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if old_path.exists() and old_path.resolve() != dst_path.resolve(): old_path.unlink(missing_ok=True)
        version_info = find_current_version_from_file(ARCHIVE_LOG_PATH(game_name))
        create_version(GAME_DIR(game_name), parent_name=version_info.get("version"), summary=f'{new_name} 교체')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: tmp_path.unlink(missing_ok=True)
        except: pass
    return JSONResponse({"status": "success", "url": f"/static/{game_name}/assets/{new_name}"})

@app.post("/regenerate-asset")
async def regenerate_asset_api(
    game_name: str = Form(...),
    asset_name: str = Form(...),
    prompt: str = Form(...)
):
    # assets의 파일명을 기반으로 asset_id (data.json index)를 찾아야 함
    game_data_path = DATA_PATH(game_name)
    game_data = {}
    if game_data_path.exists():
        with open(game_data_path, 'r', encoding='utf-8') as f:
            game_data = json.load(f)
            
    asset_id = None
    for idx, asset in enumerate(game_data.get('assets', {}).get('images', [])):
        if os.path.basename(asset.get('path', '')) == asset_name:
            asset_id = str(idx)
            break

    if asset_id:
        success, message = await _regenerate_asset_logic(game_name, asset_id, prompt)
    else:
         raise HTTPException(status_code=404, detail=f"에셋 파일 '{asset_name}'을 data.json에서 찾을 수 없습니다.")

    if success:
        return JSONResponse({
            "status": "success", 
            "url": f"/static/{game_name}/assets/{asset_name}?t={int(time.time())}"
        })
    else:
        raise HTTPException(status_code=500, detail=message)

if __name__ == "__main__":
    import uvicorn
    print("서버를 시작합니다... http://localhost:8000")
    uvicorn.run("gemini:app", host="0.0.0.0", port=8000, reload=True, workers=1)