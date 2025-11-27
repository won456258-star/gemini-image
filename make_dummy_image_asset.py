import os
import urllib.parse
import urllib.request
import random
import time
from PIL import Image, ImageDraw, ImageFont

# rembg(배경 제거) 라이브러리 확인
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    print("⚠️ 'rembg' 라이브러리가 설치되지 않았습니다. 배경 제거 기능이 비활성화됩니다.")
    REMBG_AVAILABLE = False

def check_and_create_images_with_text(data, base_directory, theme_context=""):
    """
    JSON 데이터를 기반으로 이미지를 생성합니다.
    theme_context: 사용자의 요청 내용 (예: "쿠키런 스타일로 만들어줘") -> 이것을 프롬프트에 반영합니다!
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

    print(f"\n========== [🚀 에셋 AI 자동 생성 시작 (테마: {theme_context[:20]}...)] ==========")

    for item in images_to_process:
        name = item.get('name', 'unknown')
        file_path_full = item.get('path', '')
        width = item.get('width', 64)
        height = item.get('height', 64)
        
        file_name = os.path.basename(file_path_full)
        final_save_path = os.path.join(target_directory, file_name)

        # 1. 이미 파일이 있으면 건너뜀
        if os.path.exists(final_save_path):
            continue
        
        print(f"   🎨 AI 생성 시도: {file_name} ({name})...")
        
        ai_success = False
        image_data = None

        try:
            # 🌟 [핵심 수정] 파일 이름 다듬기 & 테마 적용
            # 예: cookie_run_1 -> cookie run 1
            clean_name = name.replace("_", " ").replace("-", " ")
            
            is_background = "background" in name.lower() or "bg" in name.lower()
            
            # 🌟 [핵심 수정] 사용자의 요청(theme_context)을 프롬프트 맨 앞에 추가!
            # 예: "Cookie Run style, cookie run 1, cartoon style..."
            if theme_context:
                base_prompt = f"{theme_context} style, {clean_name}"
            else:
                base_prompt = f"{clean_name}"

            if is_background:
                prompt = f"{base_prompt}, full scenery, game background, highly detailed, no characters"
            else:
                prompt = f"{base_prompt}, game sprite, isolated object, simple white background, vector art"

            encoded_prompt = urllib.parse.quote(prompt)
            
            # 재시도 로직
            for attempt in range(1, 4):
                try:
                    seed = random.randint(0, 100000)
                    gen_width = max(512, width)
                    gen_height = max(512, height)
                    
                    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&width={gen_width}&height={gen_height}&nologo=true"
                    
                    req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
                    
                    with urllib.request.urlopen(req, timeout=60) as response:
                        image_data = response.read()
                    
                    if image_data:
                        ai_success = True
                        break 
                except Exception as e:
                    print(f"      ⚠️ 시도 {attempt} 실패: {e}")
                    time.sleep(2)

            if not ai_success or not image_data:
                raise Exception("모든 AI 생성 시도 실패")

            # 배경 제거
            if not is_background and REMBG_AVAILABLE:
                try:
                    image_data = remove(image_data)
                except Exception as e:
                    print(f"      ⚠️ 배경 제거 실패: {e}")

            with open(final_save_path, 'wb') as f:
                f.write(image_data)
            print(f"   ✨ [생성 완료] {file_name}")

        except Exception as e:
            print(f"   ❌ [실패 -> 더미 생성] {e}")
            create_dummy_image(final_save_path, width, height, name)

    print("========== [작업 완료] ==========\n")

def create_dummy_image(path, width, height, text):
    """AI 생성 실패 시 사용할 더미 이미지"""
    try:
        color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        img = Image.new('RGB', (width, height), color)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # 텍스트 중앙 정렬 (간략화)
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        img.save(path)
    except Exception as e:
        print(f"      🚨 더미 생성 실패: {e}")