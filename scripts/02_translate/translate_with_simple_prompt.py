# ultimate_final_translate.py
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

# --- 2. 궁극의 미니멀리스트 번역 프롬프트 ---
ULTIMATE_PROMPT = """Translate the following English text to natural, formal Korean.

English Text:
{text_to_translate}

Korean Translation:
"""

# --- 3. API 호출 함수들 (치환 전법 포함) ---
def api_request(messages):
    try:
        response = client.chat.completions.create(
            model="solar-1-mini-chat", messages=messages, temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"🔥🔥🔥 API 호출 중 에러 발생: {e}")
        return "" # 에러 발생 시 빈 문자열 반환

def final_translate(texts, prompt_template):
    print(f"🌐 [API Call] {len(texts)}개의 텍스트를 최종 번역합니다...")
    translated = []
    
    for text in tqdm(texts, desc="Final Korean Translation"):
        # 1. (전처리) #Person 태그를 임시 기호로 치환
        processed_text = text
        for i in range(1, 10):
            processed_text = processed_text.replace(f'#Person{i}#', f'__P{i}__')

        # 2. 번역 요청
        prompt = prompt_template.format(text_to_translate=processed_text)
        messages = [{"role": "user", "content": prompt}]
        translated_text = api_request(messages)

        # 3. (후처리) 임시 기호를 다시 #Person 태그로 복원
        for i in range(1, 10):
            translated_text = translated_text.replace(f'__P{i}__', f'#Person{i}#')
        
        translated.append(translated_text)
        
        # Rate Limiting (1초에 1개씩만 요청하여 가장 안전하게)
        time.sleep(1)
            
    return translated

# --- 4. 메인 실행 로직 ---
def main():
    print("1. 최종 영어 요약문 파일을 로드합니다...")
    input_csv_path = 'submissions/SUBMISSION_FINAL.csv'
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"🔥🔥🔥 에러: '{input_csv_path}' 파일을 찾을 수 없습니다.")
        return
        
    english_summaries = df['summary'].tolist()

    print("2. 영어 요약문을 한국어로 최종 재번역합니다...")
    korean_summaries = final_translate(english_summaries, ULTIMATE_PROMPT)

    print("3. 새로운 최종 한국어 제출 파일을 생성합니다...")
    final_df = pd.DataFrame({
        'fname': df['fname'],
        'summary': korean_summaries
    })
    output_path = 'submissions/SUBMISSION_FINAL_ULTIMATE.csv'
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n🎉🎉🎉 진짜 최종 제출 파일이 생성되었습니다: {output_path}")
    print("\n--- 최종 결과 샘플 ---")
    print(final_df.head())

if __name__ == "__main__":
    main()