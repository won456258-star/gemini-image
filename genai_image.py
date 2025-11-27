import base64
from io import BytesIO
from google.genai import types
from PIL import Image

def pil_image_to_bytes(pil_img: Image.Image, format="PNG") -> bytes:
    """PIL Image 객체를 PNG 포맷의 raw bytes로 변환합니다."""
    buffered = BytesIO()
    pil_img.save(buffered, format=format) 
    return buffered.getvalue()

def nano_banana_style_image_editing(
    gemini_client,
    model_name: str,
    reference_image: Image.Image, 
    editing_prompt: str
) -> bytes:
    """
    Gemini 모델을 사용하여 참고 이미지를 기반으로 이미지를 편집하고 raw bytes를 반환합니다.
    """
    print(f"--- 이미지 편집 요청: '{editing_prompt}' ---")
    
    try:
        # 1. PIL 이미지를 raw bytes로 변환
        input_image_bytes = pil_image_to_bytes(reference_image)
        
        # 2. 요청 구성
        contents = [
            editing_prompt, 
            types.Part.from_bytes(data=input_image_bytes, mime_type="image/png")
        ]
        
        # 3. 모델 호출
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=contents
        )
        
        # 4. 결과 추출 및 반환
        if response.candidates and response.candidates[0].content.parts:
            image_part = next((p for p in response.candidates[0].content.parts if p.inline_data), None)

            if image_part and image_part.inline_data:
                # Base64 디코딩하여 Raw Bytes 반환
                return base64.b64decode(image_part.inline_data.data)
            else:
                print("❌ 이미지 생성 결과가 응답에 포함되어 있지 않습니다.")
                return None
        else:
            print("❌ 응답 후보가 없거나 차단되었습니다.")
            return None

    except Exception as e:
        print(f"❌ 이미지 생성 중 오류 발생: {e}")
        return None