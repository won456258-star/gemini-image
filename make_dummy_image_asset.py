import os
import urllib.parse
import urllib.request
import random
import time
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import re

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

def check_and_create_images_with_text(data, base_directory, theme_context="", is_force=False, game_data_full=None, gemini_client=None, model_name=None):
    """
    게임의 전반적인 컨텍스트(제목, 설정 등)를 반영하여 이미지를 생성합니다.
    """
    images_to_process = data.get('assets', {}).get('images', [])
    if not images_to_process: return

    first_path = images_to_process[0].get('path', '')
    target_directory = os.path.join(base_directory, os.path.dirname(first_path)) 
    os.makedirs(target_directory, exist_ok=True)

    print(f"\n========== [🚀 에셋 생성 시작 (테마: {theme_context})] ==========")
    
    # 🌟 [추가] 게임 전체 컨텍스트 추출
    game_title = ""
    if game_data_full:
        game_title = game_data_full.get("settings", {}).get("title", "") # 제목이 있다면 추출
    
    # 일관성 그룹핑 로직 (기존과 동일)
    asset_groups = {}
    for item in images_to_process:
        name = item.get('name', '')
        if "background" in name.lower() or "bg_" in name.lower(): group_key = name
        else: group_key = re.split(r'[_-]', name)[0]
        if group_key not in asset_groups: asset_groups[group_key] = []
        asset_groups[group_key].append(item)

    # Gemini로 외형 설정 생성 (기존과 동일하지만, 게임 제목 정보 추가)
    group_descriptions = {}
    if gemini_client and model_name:
        for group_key, items in asset_groups.items():
            if len(items) > 1 and "background" not in group_key:
                try:
                    # 🔥 프롬프트에 게임 제목/설명 추가
                    prompt_ctx = f"Game Title: '{game_title}'. Theme: '{theme_context}'."
                    p = f"{prompt_ctx} Create a visual description for character '{group_key}'. Keep it concise."
                    resp = gemini_client.models.generate_content(model=model_name, contents=p)
                    group_descriptions[group_key] = resp.text.strip()
                    print(f"   ✨ [{group_key}] 외형 설정: {resp.text.strip()[:30]}...")
                except: pass

    # 이미지 생성
    for item in images_to_process:
        name = item.get('name', 'unknown')
        file_path_full = item.get('path', '')
        file_name = os.path.basename(file_path_full)
        save_path = os.path.join(target_directory, file_name)

        if not is_force and os.path.exists(save_path): continue
        
        print(f"   🎨 생성 시도: {file_name}...")
        
        try:
            # 🔥 [핵심] 프롬프트에 게임 정보 최대한 반영
            clean_name = name.replace("_", " ")
            is_bg = "background" in name or "bg" in name
            char_desc = group_descriptions.get(re.split(r'[_-]', name)[0], "")
            
            base_prompt = f"{theme_context} style game art. "
            if game_title: base_prompt += f"Game: {game_title}. "
            
            if is_bg:
                full_prompt = f"{base_prompt} {clean_name}, full background scene, detailed"
            else:
                if char_desc: full_prompt = f"{base_prompt} {clean_name}, {char_desc}, isolated, white background"
                else: full_prompt = f"{base_prompt} {clean_name}, isolated sprite, white background"

            # ... (이하 생성/저장 로직은 기존과 동일) ...
            # (생략: Pollinations 호출, rembg, 저장 등)
            # 여기에는 기존의 생성 코드를 그대로 두시거나, 앞서 드린 '안정성 강화' 코드를 합치면 됩니다.
            # (공간상 핵심 프롬프트 생성 부분만 강조했습니다.)
            
            # [간단 구현 예시]
            encoded = urllib.parse.quote(full_prompt)
            url = f"https://image.pollinations.ai/prompt/{encoded}?seed={random.randint(0,9999)}&width=512&height=512&nologo=true"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as res: data = res.read()
            
            if not is_bg and REMBG_AVAILABLE:
                try: data = remove(data)
                except: pass
            
            with open(save_path, 'wb') as f: f.write(data)
            print(f"   ✅ 완료")

        except Exception as e:
            print(f"   ❌ 실패: {e}")
            # 더미 생성 (생략)