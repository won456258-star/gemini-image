import urllib.parse
import urllib.request
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
    print(f"\n========== [이미지 생성 시작 (무료 모드)] ==========")
    print(f"1. 사용자 요청: {editing_prompt}")
    
    try:
        # 1. Gemini 분석 (이미지 -> 텍스트 프롬프트)
        # 구글 Gemini가 그림을 어떻게 그릴지 아주 자세한 묘사를 써줍니다.
        print(f"2. [Gemini] 이미지 분석 및 프롬프트 작성 중... (모델: {model_name})")
        input_image_bytes = pil_image_to_bytes(reference_image)
        
        analyze_prompt = f"""
        You are an expert prompt engineer. 
        User request: "{editing_prompt}"
        Based on the attached image and user's request, write a detailed English prompt for image generation.
        Focus on style, colors, and mood.
        Output ONLY the prompt text.
        """
        
        analyze_response = gemini_client.models.generate_content(
            model=model_name,
            contents=[analyze_prompt, types.Part.from_bytes(data=input_image_bytes, mime_type="image/png")]
        )
        
        generated_prompt = analyze_response.text.strip()
        print(f"   ✅ [Gemini] 프롬프트 생성 완료:\n   --> \"{generated_prompt[:100]}...\"")

        # 2. 무료 이미지 생성 (Pollinations AI 사용)
        # 결제 카드 없이 사용할 수 있는 공개 AI 서비스를 이용합니다.
        print(f"\n3. [Pollinations AI] 이미지 생성 요청 중...")
        
        # 프롬프트를 URL 주소 형식으로 변환
        encoded_prompt = urllib.parse.quote(generated_prompt)
        # 무료 생성 주소 호출 (랜덤 시드 추가로 매번 다른 그림 생성)
        import random
        seed = random.randint(0, 10000)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&width=1024&height=1024&nologo=true"
        
        # 이미지 다운로드 (파이썬 기본 라이브러리 사용)
        req = urllib.request.Request(
            image_url, 
            headers={'User-Agent': 'Mozilla/5.0'} # 웹브라우저인 척 요청
        )
        
        with urllib.request.urlopen(req) as response:
            image_data = response.read()
            
        if image_data:
            print("   ✅ [Pollinations AI] 이미지 생성 성공!")
            print("========== [작업 완료] ==========\n")
            return image_data
        else:
            print("   ❌ 응답은 받았으나 데이터가 비어있습니다.")
            return None

    except Exception as e:
        print(f"\n❌ [오류 발생]: {e}")
        return None