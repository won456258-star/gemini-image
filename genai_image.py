import urllib.parse
import urllib.request
import time
import random
from io import BytesIO
from google import genai
from google.genai import types
from PIL import Image

def pil_image_to_bytes(pil_img: Image.Image, format="PNG") -> bytes:
    buffered = BytesIO()
    pil_img.save(buffered, format=format) 
    return buffered.getvalue()

def nano_banana_style_image_editing(
    gemini_client: genai.Client,
    model_name: str, 
    reference_image: Image.Image, 
    editing_prompt: str
) -> bytes:
    print(f"\n========== [이미지 생성 시작 (안정성 모드)] ==========")
    print(f"1. 사용자 요청: {editing_prompt}")
    
    try:
        # 1. Gemini 분석 (이미지 -> 텍스트 프롬프트)
        print(f"2. [Gemini] 이미지 분석 및 프롬프트 작성 중... (모델: {model_name})")
        input_image_bytes = pil_image_to_bytes(reference_image)
        
        analyze_prompt = f"""
        You are an expert prompt engineer. 
        User request: "{editing_prompt}"
        Based on the attached image and user's request, write a detailed English prompt for image generation.
        Keep it concise (under 500 characters) to ensure stable generation.
        Focus on style, colors, and key visual elements.
        Output ONLY the prompt text.
        """
        
        analyze_response = gemini_client.models.generate_content(
            model=model_name,
            contents=[analyze_prompt, types.Part.from_bytes(data=input_image_bytes, mime_type="image/png")]
        )
        
        generated_prompt = analyze_response.text.strip()
        
        # 🌟 [안정성 패치 1] 프롬프트가 너무 길면 자르기 (URL 길이 제한 방지)
        if len(generated_prompt) > 800:
            generated_prompt = generated_prompt[:800]
            
        print(f"   ✅ [Gemini] 프롬프트 생성 완료 ({len(generated_prompt)}자)")

        # 2. 무료 이미지 생성 (Pollinations AI) - 재시도 로직 추가
        print(f"\n3. [Pollinations AI] 이미지 생성 요청 중... (최대 3회 시도)")
        
        encoded_prompt = urllib.parse.quote(generated_prompt)
        
        # 🌟 [안정성 패치 2] 3번까지 재시도하는 로직
        for attempt in range(1, 4):
            try:
                seed = random.randint(0, 100000)
                # nologo=true: 로고 제거, private=true: 비공개(선택)
                image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&width=1024&height=1024&nologo=true"
                
                req = urllib.request.Request(
                    image_url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                )
                
                # 타임아웃을 30초로 넉넉하게 설정
                with urllib.request.urlopen(req, timeout=30) as response:
                    image_data = response.read()
                
                if image_data:
                    print(f"   ✅ [Pollinations AI] 이미지 생성 성공! (시도 {attempt}회차)")
                    print("========== [작업 완료] ==========\n")
                    return image_data
            
            except Exception as e:
                print(f"   ⚠️ 시도 {attempt} 실패: {e}")
                if attempt < 3:
                    print("   ⏳ 2초 후 다시 시도합니다...")
                    time.sleep(2)
                else:
                    print("   ❌ 모든 시도 실패.")
                    return None

    except Exception as e:
        print(f"\n❌ [치명적 오류 발생]: {e}")
        return None