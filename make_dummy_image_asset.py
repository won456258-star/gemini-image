import os
import urllib.parse
import urllib.request
import random
import time
from PIL import Image, ImageDraw, ImageFont
from rembg import remove # 배경 제거 라이브러리 추가
from io import BytesIO

def check_and_create_images_with_text(data, base_directory):
    """
    JSON 데이터를 기반으로 이미지를 생성합니다.
    1. Pollinations AI로 이미지 생성 (배경 제거 포함)
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

    print(f"\n========== [초기 에셋 AI 자동 생성 시작] ==========")

    for item in images_to_process:
        name = item.get('name', 'unknown')
        file_path_full = item.get('path', '')
        width = item.get('width', 64)
        height = item.get('height', 64)
        
        # 파일명 추출 (예: cookie_run.png)
        file_name = os.path.basename(file_path_full)
        final_save_path = os.path.join(target_directory, file_name)

        # 1. 이미 파일이 있으면 건너뜀 (중요: 덮어쓰기 방지)
        if os.path.exists(final_save_path):
            print(f"   (Skip) 이미 존재함: {file_name}")
            continue
        
        # 2. AI 이미지 생성 시도
        print(f"   🎨 생성 중: {file_name} ({name})...")
        try:
            # 🌟 스타일 통일을 위한 프롬프트 설정
            # 배경이 아닌 경우 'white background'를 추가하여 배경 제거가 잘 되도록 유도
            is_background = "background" in name.lower() or "bg" in name.lower()
            
            style_tag = "cartoon style, vector art, vibrant colors, game asset"
            if is_background:
                prompt = f"{name}, {style_tag}, full scenery, highly detailed"
            else:
                prompt = f"{name}, {style_tag}, simple, white background, isolated, character sprite"

            encoded_prompt = urllib.parse.quote(prompt)
            
            # 재시도 로직 (최대 3회)
            image_data = None
            for attempt in range(1, 4):
                try:
                    seed = random.randint(0, 100000)
                    image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&width={width}&height={height}&nologo=true"
                    
                    req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=20) as response:
                        image_data = response.read()
                    
                    if image_data: break # 성공하면 루프 탈출
                except:
                    time.sleep(1) # 실패 시 1초 대기 후 재시도

            if not image_data:
                raise Exception("AI 이미지 생성 실패 (모든 시도 실패)")

            # 🌟 3. 배경 제거 로직 (캐릭터/아이템인 경우만)
            if not is_background:
                print(f"      ✂️ 배경 제거 적용 중...")
                try:
                    # rembg를 사용해 배경 제거
                    image_data = remove(image_data)
                except Exception as e:
                    print(f"      ⚠️ 배경 제거 실패 (원본 사용): {e}")

            # 파일 저장
            with open(final_save_path, 'wb') as f:
                f.write(image_data)
            print(f"   ✅ [완료] {file_name}")

        except Exception as e:
            print(f"   ⚠️ [AI 실패] 더미로 대체합니다 ({e})")
            # 실패 시 기존 더미(색깔 박스) 생성
            create_dummy_image(final_save_path, width, height, name)

    print("========== [작업 완료] ==========\n")

def create_dummy_image(path, width, height, text):
    """AI 생성 실패 시 사용할 더미 이미지 생성 함수"""
    try:
        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        img = Image.new('RGB', (width, height), color)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        # 중앙에 텍스트 대략적으로 배치 (좌표 계산 생략)
        draw.text((10, height//2 - 10), text, fill=(255, 255, 255), font=font)
        img.save(path)
        print(f"   📦 [더미] 생성 완료: {os.path.basename(path)}")
    except Exception as e:
        print(f"   ❌ 더미 생성도 실패: {e}")