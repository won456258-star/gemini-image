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
    print(f"\n========== [이미지 생성 시작 (고속 안정성 모드)] ==========")
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
        if len(generated_prompt) > 800:
            generated_prompt = generated_prompt[:800]
            
        print(f"   ✅ [Gemini] 프롬프트 생성 완료 ({len(generated_prompt)}자)")

        # 2. 무료 이미지 생성 (Pollinations AI)
        print(f"\n3. [Pollinations AI] 이미지 생성 요청 중...")
        
        encoded_prompt = urllib.parse.quote(generated_prompt)
        
        # 🌟 [최적화] 성공률을 높이기 위해 기본 크기를 512x512로 설정
        # (게임 에셋으로는 이 정도도 충분히 고화질이며, 생성 속도가 훨씬 빠릅니다)
        target_width = 512
        target_height = 512
        
        # 최대 4번 재시도
        for attempt in range(1, 5):
            try:
                seed = random.randint(0, 100000)
                # 시도 횟수가 늘어나면 크기를 더 줄여서라도 성공시키기
                if attempt > 2:
                    target_width = 256
                    target_height = 256
                    print(f"   ⚠️ (속도 향상을 위해 해상도를 {target_width}x{target_height}로 조정합니다)")

                image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&width={target_width}&height={target_height}&nologo=true"
                
                req = urllib.request.Request(
                    image_url, 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                
                # 🔥 [핵심] 타임아웃을 5분(300초)으로 설정하여 웬만해선 끊기지 않게 함
                with urllib.request.urlopen(req, timeout=300) as response:
                    image_data = response.read()
                
                if image_data:
                    print(f"   ✅ [Pollinations AI] 이미지 생성 성공! (시도 {attempt}회차)")
                    print("========== [작업 완료] ==========\n")
                    return image_data
            
            except Exception as e:
                print(f"   ⚠️ 시도 {attempt} 실패: {e}")
                if attempt < 4:
                    wait_time = attempt * 2 # 2초, 4초, 6초... 점진적 대기
                    print(f"   ⏳ {wait_time}초 후 다시 시도합니다...")
                    time.sleep(wait_time)
                else:
                    print("   ❌ 모든 시도 실패. (서버가 매우 혼잡합니다)")
                    return None

    except Exception as e:
        print(f"\n❌ [치명적 오류 발생]: {e}")
        return None