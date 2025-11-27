import base64
import os
from io import BytesIO
from google import genai
from google.genai import types
from PIL import Image
from openai import AzureOpenAI

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
    print(f"\n========== [이미지 생성 시작] ==========")
    print(f"1. 사용자 요청: {editing_prompt}")
    
    try:
        # 1. Gemini 분석
        print("2. [Gemini] 이미지 분석 중... (잠시만 기다려주세요)")
        input_image_bytes = pil_image_to_bytes(reference_image)
        
        analyze_prompt = f"""
        You are an expert DALL-E prompt engineer.
        User request: "{editing_prompt}"
        Based on the attached image and the user's request, write a detailed English prompt for DALL-E 3.
        Output ONLY the prompt text.
        """
        
        analyze_response = gemini_client.models.generate_content(
            model=model_name,
            contents=[analyze_prompt, types.Part.from_bytes(data=input_image_bytes, mime_type="image/png")]
        )
        
        generated_prompt = analyze_response.text.strip()
        print(f"   ✅ [Gemini] 프롬프트 생성 완료:\n   --> \"{generated_prompt}\"")

        # 2. Azure DALL-E 생성
        print("\n3. [Azure DALL-E] 이미지 생성 요청 중...")
        
        azure_api_key = os.getenv("AZURE_OAI_DALLE_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        # 키 확인용 (앞 5자리만 출력)
        if azure_api_key:
            print(f"   ℹ️ Azure Key 확인: {azure_api_key[:5]}... (설정됨)")
        else:
            print("   ❌ 오류: AZURE_OAI_DALLE_API_KEY가 설정되지 않았습니다!")
            return None

        azure_client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
        )

        result = azure_client.images.generate(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "dall-e-3"),
            prompt=generated_prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json"
        )

        if result.data:
            print("   ✅ [Azure DALL-E] 이미지 생성 성공!")
            print("========== [작업 완료] ==========\n")
            return base64.b64decode(result.data[0].b64_json)
        else:
            print("   ❌ [Azure DALL-E] 응답은 받았으나 이미지가 없습니다.")
            return None

    except Exception as e:
        print(f"\n❌ [치명적 오류 발생]: {e}")
        return None