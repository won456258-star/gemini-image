import os
import urllib.parse
import urllib.request
import random
import time
from PIL import Image, ImageDraw, ImageFont
# 🔥 [중요] rembg가 설치되어 있어야 합니다. (pip install rembg[gpu] 또는 pip install rembg)
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    print("⚠️ 'rembg' 라이브러리가 설치되지 않았습니다. 배경 제거 기능이 비활성화됩니다.")
    print("   설치 명령어: pip install rembg")
    REMBG_AVAILABLE = False

def check_and_create_images_with_text(data, base_directory):
    """
    JSON 데이터를 기반으로 이미지를 생성합니다.
    1. Pollinations AI로 이미지 생성 시도 (타임아웃 길게 설정)
    2. 실패 시 더미(색깔 박스) 생성
    """
    images_to_process = data.get('assets', {}).get('images', [])
    
    if images_to_process:
        first_path = images_to_process[0].get('path', '')
        target_directory = os.path.join(base_directory, os.path.dirname(first_path)) 
    else:
        return

    if not os.path.exists(target_directory):
        os.makedirs(target_directory, exist_ok=True)
        print(f"📁 디렉토리 생성: {target_directory}")

    print(f"\n========== [🚀 초기 에셋 AI 자동 생성 시작] ==========")

    for item in images_to_process:
        name = item.get('name', 'unknown')
        file_path_full = item.get('path', '')
        width = item.get('width', 64)
        height = item.get('height', 64)
        
        file_name = os.path.basename(file_path_full)
        final_save_path = os.path.join(target_directory, file_name)

        # 1. 이미 파일이 있으면 건너뜀 (중요: 덮어쓰기 방지)
        if os.path.exists(final_save_path):
            # print(f"   (Skip) 이미 존재함: {file_name}") # 너무 시끄러우면 주석 처리
            continue
        
        print(f"   🎨 AI 생성 시도: {file_name} ({name})...")
        
        ai_success = False # AI 생성 성공 여부 체크
        image_data = None

        # --- [AI 이미지 생성 시도] ---
        try:
            # 🌟 프롬프트 설정
            is_background = "background" in name.lower() or "bg" in name.lower()
            style_tag = "cartoon style, game asset, vibrant colors, cute, clean line art"
            
            if is_background:
                prompt = f"{name}, {style_tag}, full scenery, highly detailed, no characters"
            else:
                # 캐릭터/아이템은 배경 제거가 쉽도록 단순한 흰 배경 유도
                prompt = f"{name}, {style_tag}, isolated object, simple white background"

            encoded_prompt = urllib.parse.quote(prompt)
            
            # 🌟 재시도 로직 (최대 3회, 긴 타임아웃)
            for attempt in range(1, 4):
                try:
                    seed = random.randint(0, 100000)
                    # 해상도를 512 정도로 낮추면 성공률이 더 높음 (초기 에셋용으로 충분)
                    gen_width = max(512, width)
                    gen_height = max(512, height)
                    
                    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&width={gen_width}&height={gen_height}&nologo=true"
                    
                    req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
                    
                    # 🔥 [핵심 수정] 타임아웃을 60초로 설정 (AI 서버가 느릴 때를 대비)
                    with urllib.request.urlopen(req, timeout=60) as response:
                        image_data = response.read()
                    
                    if image_data:
                        ai_success = True
                        print(f"      ✅ AI 서버 응답 성공! (시도 {attempt}회차)")
                        break # 성공하면 루프 탈출
                except Exception as e:
                    print(f"      ⚠️ AI 시도 {attempt} 실패: {e}")
                    time.sleep(2) # 잠시 대기 후 재시도

            if not ai_success or not image_data:
                raise Exception("모든 AI 생성 시도 실패 (서버 혼잡 추정)")

            # 🌟 배경 제거 로직 (캐릭터/아이템인 경우만)
            if not is_background and REMBG_AVAILABLE:
                print(f"      ✂️ 배경 제거 적용 중...")
                try:
                    image_data = remove(image_data)
                except Exception as e:
                    print(f"      ⚠️ 배경 제거 실패 (원본 사용): {e}")

            # 파일 저장
            with open(final_save_path, 'wb') as f:
                f.write(image_data)
            print(f"   ✨ [AI 저장 완료] {file_name}")

        # --- [실패 시 더미 생성] ---
        except Exception as e:
            print(f"   ❌ [AI 실패] 에러 원인: {e}")
            print(f"   📦 더미(Placeholder)로 대체합니다.")
            create_dummy_image(final_save_path, width, height, name)

    print("========== [초기 에셋 생성 작업 완료] ==========\n")

def create_dummy_image(path, width, height, text):
    """AI 생성 실패 시 사용할 더미 이미지 생성 함수"""
    try:
        # 랜덤 파스텔 톤 색상
        color = (random.randint(100, 220), random.randint(100, 220), random.randint(100, 220))
        img = Image.new('RGB', (width, height), color)
        draw = ImageDraw.Draw(img)
        
        # 폰트 로드 시도 (없으면 기본 폰트)
        try:
            # 윈도우 기본 폰트 경로 예시 (시스템에 따라 다를 수 있음)
            font_path = "C:/Windows/Fonts/arial.ttf" 
            if os.path.exists(font_path):
                 font = ImageFont.truetype(font_path, size=int(min(width, height)/5))
            else:
                 font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
            
        # 텍스트 중앙 정렬 계산
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) / 2
        y = (height - text_height) / 2
        
        # 텍스트 그리기 (진한 회색)
        draw.text((x, y), text, fill=(50, 50, 50), font=font)
        
        img.save(path)
        # print(f"      (더미 파일 생성됨: {os.path.basename(path)})") # 디버그용
    except Exception as e:
        print(f"      🚨 더미 생성조차 실패: {e}")