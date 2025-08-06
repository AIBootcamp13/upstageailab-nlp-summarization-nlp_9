# final_translate.py
import pandas as pd
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# --- 1. API 클라이언트 초기화 ---
load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
client = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1/solar")
print("✅ Solar API 클라이언트가 초기화되었습니다.")

# --- 2. 챔피언 번역 프롬프트 ---
EN_TO_KO_PROMPT = """
You are a professional translator working on a Korean language dataset for AI training.
Translate the following English summary into **natural, fluent, and detailed Korean**.
**Instructions:**
1. Keep speaker tags such as `#Person1#`, `#person2#` **exactly as they are**.
2. Personal names must be translated **phonetically** (e.g., "Francis" → "프랜시스").
3. Use a **formal**, full-sentence tone.
4. DO NOT summarize or skip information. Be as detailed as the source.
---
Please translate the following English summary:
{text_to_translate}
"""

# --- 3. API 호출 함수들 (inference.py에서 가져옴) ---
def api_request_with_rate_limiting(messages):
    try:
        response = client.chat.completions.create(
            model="solar-1-mini-chat", messages=messages, temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"🔥🔥🔥 API 호출 중 에러 발생: {e}")
        return ""

def translate_texts(texts, prompt_template):
    print(f"🌐 [API Call] {len(texts)}개의 텍스트를 en -> ko 방향으로 번역합니다...")
    translated = []
    start_time = time.time()
    for i, text in enumerate(tqdm(texts, desc="Translating to Korean")):
        prompt = prompt_template.format(text_to_translate=text)
        messages = [{"role": "user", "content": prompt}]
        translated_text = api_request_with_rate_limiting(messages)
        translated.append(translated_text)
        if (i + 1) % 100 == 0 and len(texts) > 100:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time + 1
                print(f"  - Rate limit: {wait_time:.1f}초 대기...")
                time.sleep(wait_time)
            start_time = time.time()
    return translated

# --- 4. 메인 실행 로직 ---
def main():
    print("1. 영어 요약문이 담긴 제출 파일을 로드합니다...")
    input_csv_path = 'submissions/SUBMISSION_FINAL.csv'
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"🔥🔥🔥 에러: '{input_csv_path}' 파일을 찾을 수 없습니다.")
        return
        
    english_summaries = df['summary'].tolist()

    print("2. 영어 요약문을 한국어로 역번역합니다...")
    korean_summaries = translate_texts(english_summaries, EN_TO_KO_PROMPT)

    print("3. 최종 한국어 제출 파일을 생성합니다...")
    final_df = pd.DataFrame({
        'fname': df['fname'],
        'summary': korean_summaries
    })
    output_path = 'submissions/SUBMISSION_FINAL_KOREAN.csv'
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n🎉🎉🎉 진짜 최종 제출 파일이 생성되었습니다: {output_path}")
    print("\n--- 최종 결과 샘플 ---")
    print(final_df.head())

if __name__ == "__main__":
    main()