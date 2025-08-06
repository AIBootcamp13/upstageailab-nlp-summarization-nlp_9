import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm  # 진행 상황을 보여주는 라이브러리 (pip install tqdm)
import time

# --- 1. 기본 설정 ---
# 관련 라이브러리 설치: pip install python-dotenv
from dotenv import load_dotenv
import os

# .env 파일의 내용을 환경 변수로 로드합니다.
load_dotenv()

# 환경 변수에서 API 키를 불러옵니다.
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
if not UPSTAGE_API_KEY:
    raise ValueError("'.env' 파일에 UPSTAGE_API_KEY가 설정되어 있지 않습니다.")

# OpenAI 클라이언트 초기화
# base_url은 Upstage의 Solar API 엔드포인트로 설정합니다.
client = OpenAI(
    api_key=UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1/solar"
)

# 우리가 최종 선택한 '프롬프트 B'
PROMPT = """You are an expert in paraphrasing and cross-lingual adaptation. Your job is to take a Korean dialogue and rewrite it into English in a **semantically faithful but stylistically enriched** way. Your English output should retain all the **intentions, emotions, and facts**, but be phrased differently — more naturally, as a native speaker would say it in real life.

Do not translate word-for-word. Instead, reimagine the English dialogue with:
- Smoother transitions
- More idiomatic expressions
- Culturally appropriate tone
- Emotionally expressive language

Make the dialogue clear, engaging, and fully paraphrased — without losing the original meaning.

Keep speaker markers (#Person1#, #Person2#) in the output."""

# 입/출력 파일 경로 설정
# 아래의 경로가 맞나? 
# 답: 맞습니다. 왜냐하면 이 스크립트는 'train.csv' 파일을 읽어서 영어로 복원한 데이터를 
# 'train_restored_english.csv'로 저장하기 때문입니다.
INPUT_FILE = './data/raw/train.csv'
OUTPUT_FILE = './data/processed/train_restored_english.csv'

# --- 2. 핵심 기능 함수 ---

def translate_dialogue(dialogue):
    """Solar API를 호출하여 한국어 대화를 영어로 번역하는 함수"""
    try:
        response = client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=[
              {
                "role": "system",
                "content": "You are a helpful translator from Korean to English."
              },
              {
                "role": "user",
                "content": f"{PROMPT}\n\n---\n\n{dialogue}"
              }
            ],
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        # API 호출 중 에러가 발생하면, 에러 메시지를 출력하고 None을 반환
        print(f"--- API Error: {e} ---")
        return None

# --- 3. 메인 실행 로직 ---

def main():
    """전체 데이터 복원 프로세스를 실행하는 메인 함수"""
    
    # 출력 폴더가 없으면 생성
    os.makedirs('./data/processed', exist_ok=True)
    
    # 원본 데이터 불러오기
    print(f"'{INPUT_FILE}'에서 원본 데이터를 불러옵니다...")
    df = pd.read_csv(INPUT_FILE)

    # 결과를 저장할 새로운 컬럼 추가
    df['english_dialogue'] = ""
    
    print(f"총 {len(df)}개의 대화에 대한 영어 복원을 시작합니다...")
    
    # tqdm을 사용하여 진행률 표시줄 생성
    for index, row in tqdm(df.iterrows(), total=len(df)):
        
        korean_dialogue = row['dialogue']
        
        # 번역 실행
        english_translation = translate_dialogue(korean_dialogue)
        
        if english_translation:
            df.at[index, 'english_dialogue'] = english_translation
        else:
            # 번역 실패 시, 원본을 그대로 남겨두거나 특정 표시를 할 수 있음
            df.at[index, 'english_dialogue'] = "TRANSLATION_FAILED"
        
        # 100개마다 중간 결과를 저장하여 안정성 확보
        if (index + 1) % 100 == 0:
            print(f"\n... {index + 1}개 작업 완료. 중간 결과를 저장합니다 ...")
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

        # Solar API는 분당 요청 제한(Rate Limit)이 있을 수 있으므로, 
        # 약간의 지연 시간을 주어 안정성을 높임 (선택 사항)
        time.sleep(0.5) # 0.5초 대기

    # 모든 작업 완료 후 최종 저장
    print("\n모든 번역 작업을 완료했습니다. 최종 파일을 저장합니다...")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 작업 완료! 복원된 데이터가 '{OUTPUT_FILE}'에 저장되었습니다.")

# 이 스크립트가 직접 실행될 때만 main 함수를 호출
if __name__ == "__main__":
    main()