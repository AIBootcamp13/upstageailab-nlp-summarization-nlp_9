# scripts/translate_to_korean_api.py

import os
import argparse
import pandas as pd
from openai import OpenAI
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# --- 1. 기본 설정 ---
# .env 파일에서 API 키를 불러오도록 설정하는 것이 안전함
from dotenv import load_dotenv
load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
if not UPSTAGE_API_KEY:
    raise ValueError("UPSTAGE_API_KEY 환경 변수를 설정해주세요.")

client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)


PROMPT_KOREAN = """You are a professional translator for a machine learning competition.
Translate the following English summary into natural, fluent Korean.

**Rules:**
1.  **Preserve Speaker Markers**: Tokens like `#Person1#`, `#person2#` must be kept exactly as they are. DO NOT translate them into `#사람1#` or anything else.
2.  **Phonetic Names**: Translate personal names based on their sound (phonetically). DO NOT guess or change them into other famous figures. For example, "Francis" should be translated as "프랜시스", not "프란치스코 교황".
3.  **Formal Tone**: The translation must be clear and objective, in a formal tone, as if written by a human annotator.

**Example:**

[Input English Summary]:
"#Person1# inquired with #person2# about the necessary terminology when purchasing shoes, and #person2# provided valuable information in response."

[Output Korean Summary]:
"#Person1#이 #person2#에게 신발 구매 시 필요한 용어에 대해 문의하자, #person2#는 유용한 정보를 제공했다."

---
**English Text to Translate:**
"""
def translate_text(text):
    """Solar API를 호출하여 텍스트를 번역하는 함수 (재시도 기능 포함)"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    max_retries = 5
    delay = 5
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="solar-1-mini-chat",
                messages=[{"role": "user", "content": f"{PROMPT_KOREAN}\n\n---\n\n{text}"}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            if '429' in str(e):
                print(f"Rate limit hit. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"API Error: {e} for text: {text[:30]}...")
                return "TRANSLATION_FAILED"
    return "TRANSLATION_FAILED"


# scripts/translate_to_korean_api.py 의 main 함수를 교체

def main(input_file_path):
    print("🚀 Solar API를 사용한 최종 한국어 번역을 시작합니다...")

    df_english = pd.read_csv(input_file_path)
    english_summaries = df_english['summary'].tolist()
    print(f"'{input_file_path}' 파일에서 {len(english_summaries)}개의 요약문을 읽었습니다.")

    korean_summaries = ["" for _ in english_summaries]
    processed_count = 0
    
    # --- 수정된 부분 시작 ---
    submissions_dir = "submissions"
    os.makedirs(submissions_dir, exist_ok=True)
    kst = timezone(timedelta(hours=9))
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    
    # 중간 저장용 파일 경로를 루프 밖에서 한 번만 정의
    checkpoint_path = os.path.join(submissions_dir, f'submission_korean_api_{timestamp}_checkpoint.csv')

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_index = {executor.submit(translate_text, text): i for i, text in enumerate(english_summaries)}
        
        for future in tqdm(as_completed(future_to_index), total=len(english_summaries)):
            index = future_to_index[future]
            try:
                korean_summaries[index] = future.result()
            except Exception as e:
                print(f"번역 작업 중 에러 발생 (인덱스 {index}): {e}")
                korean_summaries[index] = "TRANSLATION_FAILED"
            
            processed_count += 1
            
            # 30개마다 중간 저장 (하나의 파일에 덮어쓰기)
            if processed_count % 30 == 0:
                print(f"\n... {processed_count}개 번역 완료. 중간 저장합니다 ...")
                df_checkpoint = pd.DataFrame({
                    'fname': df_english['fname'],
                    'summary': korean_summaries
                })
                # 항상 동일한 체크포인트 파일에 덮어쓰기
                df_checkpoint.to_csv(checkpoint_path, index=True, encoding='utf-8-sig')

    # --- 수정된 부분 끝 ---

    print("✅ 번역이 완료되었습니다.")

    # 최종 제출 파일 생성
    df_korean = pd.DataFrame({
        'fname': df_english['fname'],
        'summary': korean_summaries
    })
    
    submission_path = os.path.join(submissions_dir, f'submission_korean_api_{timestamp}_final.csv')
    df_korean.to_csv(submission_path, index=True, encoding='utf-8-sig')
    print(f"🎉 최종 제출 파일이 '{submission_path}'에 성공적으로 저장되었습니다.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="번역할 영어 요약문이 담긴 CSV 파일 경로")
    args = parser.parse_args()
    main(args.input_file)