import base64
import os
from io import BytesIO
from google import genai
from google.genai import types
from PIL import Image
from openai import AzureOpenAI  # Azure 클라이언트 임포트

def pil_image_to_bytes(pil_img: Image.Image, format="PNG") -> bytes:
    """PIL Image 객체를 PNG 포맷의 raw bytes로 변환합니다."""
    buffered = BytesIO()
    pil_img.save(buffered, format=format) 
    return buffered.getvalue()

def nano_banana_style_image_editing(
    gemini_client: genai.Client,
    model_name: str, 
    reference_image: Image.Image, 
    editing_prompt: str
) -> bytes:
    """
    [기능]
    1. Gemini (Vision): 원본 이미지를 분석하여 DALL-E 3용 영어 프롬프트 작성
    2. Azure DALL-E 3: 실제 이미지 생성
    """
    print(f"--- [1단계] Gemini: 이미지 분석 및 DALL-E 프롬프트 작성 중... ---")
    
    try:
        # 1. 이미지를 Bytes로 변환 (Gemini 전송용)
        input_image_bytes = pil_image_to_bytes(reference_image)
        
        # 2. Gemini에게 이미지 설명을 요청 (Vision 기능)
        analyze_prompt = f"""
        You are an expert DALL-E prompt engineer.
        User request: "{editing_prompt}"
        
        Based on the attached image and the user's request, write a detailed English prompt for DALL-E 3 to generate a new image.
        Describe the style, subject, colors, and composition in detail.
        Output ONLY the prompt text.
        """
        
        # gemini.py에서 전달받은 클라이언트와 모델(gemini-1.5-flash) 사용
        analyze_response = gemini_client.models.generate_content(
            model=model_name, 
            contents=[
                analyze_prompt,
                types.Part.from_bytes(data=input_image_bytes, mime_type="image/png")
            ]
        )
        
        generated_prompt = analyze_response.text.strip()
        print(f"🤖 Gemini가 생성한 프롬프트: {generated_prompt}")

        # 3. Azure OpenAI (DALL-E 3)로 이미지 생성
        print(f"--- [2단계] Azure DALL-E: 이미지 생성 중... ---")
        
        # Azure 설정 로드
        azure_api_key = os.getenv("AZURE_OAI_DALLE_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "dall-e-3")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not azure_api_key or not azure_endpoint:
            print("❌ 오류: .env 파일에 Azure API 설정이 없습니다.")
            return None

        # Azure 클라이언트 초기화
        azure_client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
        )

        # 이미지 생성 요청
        result = azure_client.images.generate(
            model=azure_deployment,
            prompt=generated_prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json"
        )

        # 4. 결과 반환
        if result.data:
            image_b64 = result.data[0].b64_json
            return base64.b64decode(image_b64)
        else:
            print("❌ Azure DALL-E가 이미지를 반환하지 않았습니다.")
            return None

    except Exception as e:
        print(f"❌ 이미지 처리 중 오류 발생: {e}")
        return None