import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm  # 진행 상황을 보여주는 라이브러리 (pip install tqdm)
import time
from dotenv import load_dotenv
import os

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

# 1. 프롬프트 수정: 이제는 대화가 아닌 '요약문'을 번역하므로, 더 간단한 프롬프트로 변경
# PROMPT = "Translate the following Korean summary into a natural and fluent English summary:"
PROMPT = """Rewrite the following Korean summary in fluent, natural English for use in a summarization model.  
Do not translate word-for-word. Instead:
- Use English summarization style
- Reorganize sentences for clarity
- Remove redundant words
- Maintain factual accuracy
"""

# 2. 입/출력 파일 경로 수정
# 입력 파일은 방금 만든 영어 복원 파일
INPUT_FILE = './data/processed/train_restored_english.csv' 
# 최종 결과물이 저장될 파일
OUTPUT_FILE = './data/processed/train_final_english.csv'


# 입/출력 파일 경로 설정
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
    
    # --- 경로 디버깅 코드 ---
    # 스크립트가 실행되는 현재 위치를 출력
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    
    # ▼▼▼▼▼ 핵심 수정 부분 ▼▼▼▼▼
    # 프로젝트의 루트 디렉토리 경로를 찾음 (scripts 폴더의 부모 폴더)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 절대 경로를 사용하여 입/출력 파일 경로를 명확히 지정
    input_file_path = os.path.join(project_root, 'data', 'processed', 'train_restored_english.csv')
    output_file_path = os.path.join(project_root, 'data', 'processed', 'train_final_english.csv')
    
    # 저장하려는 파일의 전체 경로를 출력해서 확인
    print(f"입력 파일 경로: {input_file_path}")
    print(f"출력 파일 경로: {output_file_path}")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # 출력 폴더가 없으면 생성
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # 원본 데이터 불러오기
    print(f"'{input_file_path}'에서 원본 데이터를 불러옵니다...")
    df = pd.read_csv(input_file_path)

    df['english_summary'] = ""
    
    print(f"총 {len(df)}개의 요약문에 대한 영어 번역을 시작합니다...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        korean_summary = row['summary']
        english_translation = translate_dialogue(korean_summary)
        
        if english_translation:
            df.at[index, 'english_summary'] = english_translation
        else:
            df.at[index, 'english_summary'] = "TRANSLATION_FAILED"
        
        if (index + 1) % 100 == 0:
            print(f"\n... {index + 1}개 작업 완료. 중간 결과를 저장합니다 ...")
            # 절대 경로로 저장
            df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

        time.sleep(0.5)

    print("\n모든 번역 작업을 완료했습니다. 최종 파일을 저장합니다...")
    # 절대 경로로 최종 저장
    df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 작업 완료! 최종 데이터가 '{output_file_path}'에 저장되었습니다.")


if __name__ == "__main__":
    main()