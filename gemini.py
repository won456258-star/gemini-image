import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import io 
import os
from dotenv import load_dotenv

# --- 추가된 라이브러리 ---
from fastapi import Response, File, UploadFile, Form, HTTPException, Query, Request, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from PIL import Image
from rembg import remove # 배경 제거
from genai_image import nano_banana_style_image_editing # 이미지 생성 함수
from realtime import List
import ffmpeg

# 기존 모듈 임포트 유지
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
    "http://localhost:3000",
    "http://localhost:8080",
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

# [Gemini 설정] 채팅 및 이미지 분석용
gemini_api_key = os.getenv('GEMINI_API_KEY')
model_name = "gemini-2.5-flash"  # 채팅/코드 수정용 모델

# Gemini 클라이언트 초기화
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

# -------------------------------------------------------------------------
#  [기존 유틸리티 함수들] - 코드 수정/파싱 등 (그대로 유지)
# -------------------------------------------------------------------------

def remove_comments_from_file(file_path):
    if not os.path.exists(file_path):
        return ""
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

# (경로 관련 함수들)
def GAME_DIR(game_name:str): return BASE_PUBLIC_DIR / game_name
def CODE_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "game.ts"
def DATA_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "data.json"
def SPEC_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "spec.md"
def CHAT_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "chat.json"
def ASSETS_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "assets"
def ARCHIVE_LOG_PATH(game_name:str): return BASE_PUBLIC_DIR / game_name / "archive" / "change_log.json"
CODE_PATH_NOCOMMENT = "" 

# (JSON 파싱 함수들)
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

# -------------------------------------------------------------------------
#  [기존 API 엔드포인트들] - 채팅, 코드 수정 등 (그대로 유지)
# -------------------------------------------------------------------------

def modify_code(message, question, game_name):
    create_project_structure(GAME_DIR(game_name))
    original_code = ""
    if os.path.exists(CODE_PATH(game_name)):
        with open(CODE_PATH(game_name), 'r', encoding='utf-8') as f:
            original_code = f.read()
    original_data = ""
    if os.path.exists(DATA_PATH(game_name)):
        with open(DATA_PATH(game_name), 'r', encoding='utf-8') as f:
            original_data = f.read()

    request_obj = type('obj', (object,), {'message': message, 'game_name': game_name})
    
    if original_code == "":
        prompt = makePTP.get_final_prompt(request_obj, question)
    else:
        prompt = modifyPTP.get_final_prompt(request_obj, question, original_code, original_data)

    print(f"AI 모델이 작업 중 입니다: {model_name}...")
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    responseData = parse_ai_code_response(response.text)
    game_code = remove_code_fences_safe(responseData.get('game_code', ''))
    game_data = remove_code_fences_safe(responseData.get('game_data', ''))
    description = remove_code_fences_safe(responseData.get('description', ''))

    modify_check = ""
    if game_code and game_code != '':
        directory_path = os.path.dirname(CODE_PATH(game_name)) 
        if directory_path: os.makedirs(directory_path, exist_ok=True)
        with open(CODE_PATH(game_name), 'w', encoding='utf-8') as f: f.write(game_code)
        modify_check = "< game.ts : 수정 O >   "
    else:
        modify_check = "< game.ts : 수정 X >   "

    error = ""
    if game_data and game_data != '':    
        error = validate_json(game_data)
        json_data = json.loads(game_data)
        check_and_create_images_with_text(json_data, GAME_DIR(game_name))
        copy_and_rename_sound_files(json_data, GAME_DIR(game_name))
        directory_path = os.path.dirname(DATA_PATH(game_name)) 
        if directory_path: os.makedirs(directory_path, exist_ok=True)
        with open(DATA_PATH(game_name), 'w', encoding='utf-8') as f: f.write(game_data)
        modify_check += "< data.json : 수정 O >\n"
    else:
        modify_check += "< data.json : 수정 X >\n"

    description = modify_check + description
    
    if error == "":
        error = check_typescript_compile_error(CODE_PATH(game_name))
    else:
        error = error + '\n' + check_typescript_compile_error(CODE_PATH(game_name))

    return game_code, game_data, description, error

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
    당신은 사용자쿼리가 아래의 카테고리 중 어디에 속하는지 분류해야 합니다.
        1: 게임을 수정해 달라는 요청.
        2: 게임과 관련된 질문.
        3: 기타.
        4: 부적절/비윤리적/서비스 범위초과
    아래와 같은 json 형식으로 답변해 주세요.
    {
        "category": int,
        "dscription: str,
        "response": str
    }
    """
    response = gemini_client.models.generate_content(model=model_name, contents=prompt)
    reply_content = json.loads(remove_code_fences_safe(response.text))
    cat = reply_content['category']
    
    result_text = ""
    if cat == 1:
        # process-code에서 처리하므로 여기선 간단한 응답만 하거나 로직 분리 필요. 
        # 원본 로직 유지: modify_code 호출
        _, _, _, _ = modify_code(request.message, "", request.game_name) # 임시 호출 (실제로는 process-code가 메인)
        result_text = "수정되었습니다."
    elif cat == 2:
        result_text = describe_code(request)
    elif cat == 4:
        result_text = "제가 도와드릴 수 없는 요청이에요."
    
    return {"status": "success", "reply": result_text}

@app.post("/process-code")
async def process_code(request: CodeRequest):
    game_name = request.game_name
    prompt = pdp.get_final_prompt(request.message)
    
    success = False
    fail_message = ""
    for i in range(5):    
        try:
            response = gemini_client.models.generate_content(model=model_name, contents=prompt)
            success = True
            break
        except Exception as e: fail_message = f"❌ 에러 발생: {e}"

    if not success:
        save_chat(CHAT_PATH(game_name), "bot", fail_message)
        return {"status": "fail", "reply": fail_message}

    devide = json.loads(remove_code_fences_safe(response.text))
    Modification_Requests = devide.get("Modification_Requests", [])
    Questions = devide.get("Questions", [])
    Inappropriate = devide.get("Inappropriate", [])
    
    Inappropriate_answer = ""
    if len(Inappropriate) > 0:
        formatted_lines = [f"죄송합니다 '{item}'는 도와드릴 수 없습니다." for item in Inappropriate]
        Inappropriate_answer = "\n\n" + "\n".join(formatted_lines)

    user_requests = "\n".join(Modification_Requests)
    user_question = "\n".join(Questions)
    devide_result = f"요청:\n{user_requests}\n질문:\n{user_question}\n부적절:\n{Inappropriate_answer}\n"
    print(devide_result)

    # 1. 질문만 있는 경우
    if len(Modification_Requests) == 0: 
        save_chat(CHAT_PATH(game_name), "user", request.message)       
        if len(Questions) == 0:
            return {"status": "success", "reply": devide_result + Inappropriate_answer + "\n\n무엇을 도와드릴까요?"}
        else:
            original_code = ""
            if os.path.exists(CODE_PATH(game_name)):
                with open(CODE_PATH(game_name), 'r', encoding='utf-8') as f: original_code = f.read()
            original_data = ""
            if os.path.exists(DATA_PATH(game_name)):
                with open(DATA_PATH(game_name), 'r', encoding='utf-8') as f: original_data = f.read()

            q_prompt = qtp.get_final_prompt(user_question, original_code, original_data)
            answer = ""
            try:
                response = gemini_client.models.generate_content(model=model_name, contents=q_prompt)
                answer = parse_ai_answer_response(response.text)['answer']
            except Exception as e:
                fail_message = f"❌ 에러 발생: {e}"
                save_chat(CHAT_PATH(game_name), "bot", fail_message)
                return {"status": "fail", "reply": fail_message}

            answer = devide_result + answer + "\n\n" + Inappropriate_answer
            save_chat(CHAT_PATH(game_name), "bot", answer)
            return {"status": "success", "reply": answer}
    
    # 2. 수정 요청이 있는 경우
    else:
        is_first_created = not os.path.exists(CODE_PATH(game_name))
        try:
            save_chat(CHAT_PATH(game_name), "user", user_requests)
            game_code, game_data, description_total = "", "", ""
            success = False
            
            for i in range(5):    
                try:
                    game_code, game_data, description, error = modify_code(user_requests, user_question, game_name) 
                    description_total += description
                    if error == "":
                        success = True
                        break 
                    else:
                        user_requests = error # 에러 발생 시 에러 내용을 다음 프롬프트로 사용
                        description_total += f"\n\n========Compile Error========\n{error}\n=============================\n"
                except Exception as e:     
                    print(f"❌ 에러 발생: {e}")
                
                user_question = "" # 에러 수정 시 질문은 제거

            if success:
                if game_code != '' or game_data != '':
                    if is_first_created:
                        create_version(GAME_DIR(game_name), summary=user_requests)
                    else:
                        version_info = find_current_version_from_file(ARCHIVE_LOG_PATH(game_name))
                        current_ver = version_info.get("version")
                        create_version(GAME_DIR(game_name), parent_name=current_ver, summary=user_requests)
                        
                description_total = devide_result + description_total + "\n\n" + Inappropriate_answer
                save_chat(CHAT_PATH(game_name), "bot", description_total)
                return {"status": "success", "code": game_code, "data": game_data, "reply": description_total}
            else:                
                fail_message = devide_result + description_total + "\n\n" + Inappropriate_answer
                save_chat(CHAT_PATH(game_name), "bot", fail_message)
                return {"status": "fail", "reply": fail_message}
        except Exception as e:            
            save_chat(CHAT_PATH(game_name), "bot", "서버오류: " + str(e))
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/spec")
async def get_spec(game_name: str):
    spec = " "
    if os.path.exists(SPEC_PATH(game_name)):
        with open(SPEC_PATH(game_name), 'r', encoding='utf-8') as f: spec = f.read()
    return spec

@app.get("/game_data")
async def get_game_data(game_name: str):
    if os.path.exists(DATA_PATH(game_name)):
         with open(DATA_PATH(game_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
    else: return {}
    return data

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

class RestoreRequest(BaseModel):
    version: str
    game_name: str

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

@app.get("/load-chat")
def load_chat_request(game_name: str = Query(..., min_length=1)):
    try: return load_chat(CHAT_PATH(game_name))
    except Exception: return {"chat": []}

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

@app.post("/client-error")
async def receive_client_error(batch: ErrorBatch):
    print(batch.error_report)
    save_chat(CHAT_PATH(batch.game_name), "bot", batch.error_report)
    return {"status": "success"}

class DataUpdatePayload(BaseModel):
    game_name: str
    data: dict

@app.post("/data-update")
async def data_update(update: DataUpdatePayload):
    with open(DATA_PATH(update.game_name), 'w', encoding='utf-8') as f:
        json.dump(update.data, f, ensure_ascii=False, indent=4)
    version_info = find_current_version_from_file(ARCHIVE_LOG_PATH(update.game_name))
    create_version(GAME_DIR(update.game_name), parent_name=version_info.get("version"), summary='게임 데이터 수정')
    return {"status": "success"}

class WrappedSubmitData(BaseModel):
    game_name: str
    payload: str

@app.post("/qna")
async def qna_process(data: WrappedSubmitData):
    game_name = data.game_name
    chat_data = json.loads(data.payload)
    
    # format_json_to_string 로직 (간소화)
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

    prompt = sqtp.get_final_prompt("", "", spec) # history 비움
    response = gemini_client.models.generate_content(model=model_name, contents=prompt)
    
    return {"status": "success", "reply": remove_code_fences_safe(response.text)}

# RevertRequest 클래스 정의 추가
class RevertRequest(BaseModel):
    game_name: str

@app.post("/revert")
async def revert_code(request: RevertRequest):
    version_info = find_current_version_from_file(ARCHIVE_LOG_PATH(request.game_name))
    restore_success = restore_version(GAME_DIR(request.game_name), version_info.get("parent"))
    if restore_success:
        save_chat(CHAT_PATH(request.game_name), "bot", "코드를 이전 버전으로 되돌렸습니다.")
        return {"status": "success", "reply": "되돌리기 성공"}
    else:
        return {"status": "success", "reply": "되돌릴 내역 없음"}

# -------------------------------------------------------------------------
#  [신규 기능] 이미지 생성(Azure DALL-E) 및 배경 제거(rembg) API
# -------------------------------------------------------------------------

GAMES_ROOT_DIR = BASE_PUBLIC_DIR.resolve() 

@app.post("/generate-image")
async def generate_image_api(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    """
    1. Gemini(Vision)로 이미지를 분석 (gemini-1.5-flash)
    2. 분석된 내용을 바탕으로 Azure DALL-E 3가 이미지를 생성
    """
    vision_model_name = "gemini-1.5-flash" 

    try:
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        result_bytes = nano_banana_style_image_editing(
            gemini_client=gemini_client,
            model_name=vision_model_name,
            reference_image=pil_image,
            editing_prompt=prompt
        )

        if result_bytes:
            return Response(content=result_bytes, media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail="이미지 생성 실패")

    except Exception as e:
        print(f"API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/remove-bg")
async def remove_background_api(image: UploadFile = File(...)):
    """
    rembg 라이브러리를 사용하여 배경 제거
    """
    try:
        image_data = await image.read()
        result_data = remove(image_data)
        return Response(content=result_data, media_type="image/png")
    except Exception as e:
        print(f"배경 제거 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class AssetItem(BaseModel):
    name: str
    url: str
class AssetsResponse(BaseModel):
    images: List[AssetItem]
    sounds: List[AssetItem]

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

        if old_path.exists() and old_path.resolve() != dst_path.resolve():
            old_path.unlink(missing_ok=True)
        
        version_info = find_current_version_from_file(ARCHIVE_LOG_PATH(game_name))
        create_version(GAME_DIR(game_name), parent_name=version_info.get("version"), summary=f'{new_name} 교체')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: tmp_path.unlink(missing_ok=True)
        except: pass
                 
    return JSONResponse({"status": "success", "url": f"/static/{game_name}/assets/{new_name}"})

if __name__ == "__main__":
    import uvicorn
    print("서버를 시작합니다... http://localhost:8000")
    uvicorn.run("gemini:app", host="0.0.0.0", port=8000, reload=True, workers=1)