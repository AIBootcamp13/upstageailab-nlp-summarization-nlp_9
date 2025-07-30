import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
from dotenv import load_dotenv

''' 
기존 전략 (폐기) : <health_checkup> 같은 스페셜 토큰으로 만든다.

새로운 전략 (채택) : 번역된 topic을 자연스러운 영어 문장 형태로 대화문 앞에 붙여서, 모델에게 더 풍부한 '문맥 힌트'를 준다.

예시: topic: Health checkup. Dialogue: #Person1#: Hello, Mr. Smith...'''


# .env 파일의 내용을 환경 변수로 로드합니다.
load_dotenv()

# 환경 변수에서 API 키를 불러옵니다.
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

if not UPSTAGE_API_KEY:
    raise ValueError("'.env' 파일에 UPSTAGE_API_KEY가 설정되어 있지 않습니다.")


client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)


# Topic 번역을 위한 프롬프트 설정
# PROMPT = "Translate the following Korean topic title into a concise and natural English topic title:"
PROMPT = (
    "You are a professional topic title translator.\n"
    "Translate the following Korean topic title into a clear, concise, and natural English topic title.\n"
    "Avoid overtranslation. Use simple everyday English. Do NOT add extra explanation.\n"
    "Korean Title:"
)


# 입/출력 파일 경로 설정
INPUT_FILE = './data/processed/train_final_english.csv'
OUTPUT_FILE = './data/processed/train_dataset_final.csv'

# --- 2. 핵심 기능 함수 ---
def translate_text(text):
    """Solar API를 호출하여 텍스트를 번역하는 범용 함수"""
    try:
        response = client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=[
                {"role": "user", "content": f"{PROMPT}\n\n---\n\n{text}"}
            ],
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"--- API Error: {e} ---")
        return None

# --- 3. 메인 실행 로직 ---
def main():
    print(f"'{INPUT_FILE}'에서 데이터를 불러옵니다...")
    df = pd.read_csv(INPUT_FILE)

    # 결과를 저장할 새로운 컬럼 추가
    df['english_topic'] = ""
    
    print(f"총 {len(df)}개의 Topic에 대한 영어 번역을 시작합니다...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        korean_topic = row['topic']
        english_translation = translate_text(korean_topic)
        
        if english_translation:
            df.at[index, 'english_topic'] = english_translation
        else:
            df.at[index, 'english_topic'] = "TRANSLATION_FAILED"
        
        # 중간 저장은 이번엔 생략. Topic은 짧아서 금방 끝날 거야.
        # time.sleep(0.5) # Rate Limit이 걱정되면 주석 해제

    print("\n모든 번역 작업을 완료했습니다. 최종 파일을 저장합니다...")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 작업 완료! 최종 데이터가 '{OUTPUT_FILE}'에 저장되었습니다.")

if __name__ == "__main__":
    main()