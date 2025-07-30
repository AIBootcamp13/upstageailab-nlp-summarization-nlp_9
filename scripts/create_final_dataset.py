import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split

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

# Dialogue 번역을 위한 '우승 프롬프트 B'
PROMPT_DIALOGUE = """You are an expert in paraphrasing and cross-lingual adaptation. Your job is to take a Korean dialogue and rewrite it into English in a **semantically faithful but stylistically enriched** way. Your English output should retain all the **intentions, emotions, and facts**, but be phrased differently — more naturally, as a native speaker would say it in real life. Do not translate word-for-word. Instead, reimagine the English dialogue with smoother transitions, more idiomatic expressions, culturally appropriate tone, and emotionally expressive language. Make the dialogue clear, engaging, and fully paraphrased — without losing the original meaning. Keep speaker markers (#Person1#, #Person2#) in the output."""

# Summary 번역을 위한 '우승 프롬프트 C'
PROMPT_SUMMARY = "Translate the Korean dialogue into a natural but informative English style, maintaining key details and avoiding casual expressions. The result should be clear and objective, as if written by a human annotator."

# 입/출력 파일 경로 설정
INPUT_FILE = './data/raw/train.csv'
CHECKPOINT_FILE = './data/processed/translation_checkpoint.csv'
FINAL_TRAIN_FILE = './data/processed/train.csv'
FINAL_VAL_FILE = './data/processed/val.csv'

# --- 2. 핵심 기능 함수 ---

def translate(text, prompt):
    """Solar API를 호출하여 텍스트를 번역하는 범용 함수"""
    try:
        response = client.chat.completions.create(
            model="solar-1-mini-chat",
            messages=[
                {"role": "user", "content": f"{prompt}\n\n---\n\n{text}"}
            ],
            stream=False,
            temperature=0.1 # 일관된 번역을 위해 temperature를 낮춤
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"--- API Error: {e} ---")
        return "TRANSLATION_FAILED"

# --- 3. 메인 실행 로직 ---

def main():
    """전체 데이터 생성 파이프라인을 실행하는 메인 함수"""
    
    os.makedirs('./data/processed', exist_ok=True)
    
    start_index = 0
    if os.path.exists(CHECKPOINT_FILE):
        print("💾 중간 저장된 체크포인트 파일을 불러옵니다...")
        df = pd.read_csv(CHECKPOINT_FILE)
        # 마지막으로 작업한 행 다음부터 시작
        last_done_index = df['english_dialogue'].last_valid_index()
        if last_done_index is not None:
            start_index = last_done_index + 1
    else:
        print(f"'{INPUT_FILE}'에서 원본 데이터를 불러옵니다...")
        df = pd.read_csv(INPUT_FILE)
        df['english_dialogue'] = None # 새 컬럼 생성
        df['english_summary'] = None  # 새 컬럼 생성

    print(f"🚀 총 {len(df)}개의 데이터 중, {start_index}번부터 번역을 시작합니다...")

    for index in tqdm(range(start_index, len(df))):
        # Dialogue 번역
        if pd.isnull(df.at[index, 'english_dialogue']):
            korean_dialogue = df.at[index, 'dialogue']
            df.at[index, 'english_dialogue'] = translate(korean_dialogue, PROMPT_DIALOGUE)

        # Summary 번역
        if pd.isnull(df.at[index, 'english_summary']):
            korean_summary = df.at[index, 'summary']
            df.at[index, 'english_summary'] = translate(korean_summary, PROMPT_SUMMARY)
        
        # 50개마다 중간 결과를 저장하여 안정성 확보
        if (index + 1) % 50 == 0:
            print(f"\n... {index + 1}개 작업 완료. 체크포인트를 저장합니다 ...")
            df.to_csv(CHECKPOINT_FILE, index=False, encoding='utf-8-sig')

        time.sleep(0.5)

    print("\n✅ 모든 번역 작업을 완료했습니다.")
    
    # --- 최종 전처리 및 저장 ---
    print("🧹 최종 데이터 전처리 및 분할을 시작합니다...")
    
    # 데이터 정제
    df = df.dropna()
    df = df[~df['english_dialogue'].str.contains("TRANSLATION_FAILED", na=False)]
    df = df[~df['english_summary'].str.contains("TRANSLATION_FAILED", na=False)]
    
    # 소문자화
    df['english_dialogue'] = df['english_dialogue'].str.lower()
    df['english_summary'] = df['english_summary'].str.lower()
    
    # 최종적으로 필요한 컬럼만 선택
    final_df = df[['english_dialogue', 'english_summary', 'topic']]

    # 데이터셋 분할
    train_df, val_df = train_test_split(final_df, test_size=0.1, random_state=42, shuffle=True)
    
    # 최종 파일 저장
    train_df.to_csv(FINAL_TRAIN_FILE, index=False, encoding='utf-8-sig')
    val_df.to_csv(FINAL_VAL_FILE, index=False, encoding='utf-8-sig')

    print("\n🎉🎉🎉 모든 작업 완료! 최종 훈련/검증 데이터가 아래 경로에 저장되었습니다.")
    print(f"  - 훈련셋: {FINAL_TRAIN_FILE} ({len(train_df)}개)")
    print(f"  - 검증셋: {FINAL_VAL_FILE} ({len(val_df)}개)")
    
    # 중간 파일 삭제 (선택 사항)
    # if os.path.exists(CHECKPOINT_FILE):
    #     os.remove(CHECKPOINT_FILE)

if __name__ == "__main__":
    main()