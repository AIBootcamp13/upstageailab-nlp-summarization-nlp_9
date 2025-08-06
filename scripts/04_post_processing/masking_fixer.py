
import pandas as pd
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import re

# --- 1. API 클라이언트 초기화 및 프롬프트 정의 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("UPSTAGE_API_KEY"), base_url="https://api.upstage.ai/v1/solar")
ULTIMATE_PROMPT = "Translate the following English text to natural, formal Korean.\n\nEnglish Text:\n{text_to_translate}\n\nKorean Translation:"
print("✅ API 클라이언트 및 프롬프트가 준비되었습니다.")

# --- 2. '마스킹'을 위한 이름 및 태그 사전 ---
# 여기에 우리가 아는 모든 이름과 태그를 '비밀 코드'와 함께 정의
MASK_MAP = {
    "#Person1#": "__P1__",
    "#Person2#": "__P2__",
    "#Person3#": "__P3__",
    "#Person4#": "__P4__",
    "#Person5#": "__P5__",
    "#Person6#": "__P6__",
    "#Person7#": "__P7__",
    "Tom": "__TOM__",
    "Brian": "__BRIAN__",
    "Kate": "__KATE__",
    "Masha": "__MASHA__",
    "Hero": "__HERO__",
    "John": "__JOHN__",
    "Francis": "__FRANCIS__",
    "Steven": "__STEVEN__",
    "Tony": "__TONY__",
    "Rose": "__ROSE__",
    "Jack": "__JACK__",
    "Mike": "__MIKE__",
    "Dawson": "__DAWSON__",
    "Maya": "__MAYA__",
    "James": "__JAMES__",
    "Muriel": "__MURIEL__",
    "L.A.": "__LA__",
    # ... 필요 시 더 추가 ...
}
# 언마스킹을 위해 key-value를 뒤집은 사전도 준비
UNMASK_MAP = {v: k for k, v in MASK_MAP.items()}

def mask_text(text):
    for real, placeholder in MASK_MAP.items():
        text = text.replace(real, placeholder)
    return text

def unmask_text(text):
    for placeholder, real in UNMASK_MAP.items():
        text = text.replace(placeholder, real)
    return text

# --- 3. API 호출 및 번역 함수 ---
def api_request(messages):
    try:
        response = client.chat.completions.create(model="solar-1-mini-chat", messages=messages, temperature=0.1)
        return response.choices[0].message.content
    except Exception as e:
        print(f"🔥🔥🔥 API 호출 중 에러 발생: {e}")
        return ""

def final_translate(texts, prompt_template):
    translated = []
    for text in tqdm(texts, desc="Final Korean Translation"):
        masked_text = mask_text(str(text))
        prompt = prompt_template.format(text_to_translate=masked_text)
        messages = [{"role": "user", "content": prompt}]
        translated_masked_text = api_request(messages)
        unmasked_text = unmask_text(translated_masked_text)
        translated.append(unmasked_text)
        time.sleep(1) # Rate Limiting
    return translated

import pandas as pd
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import re

# --- 1. API 클라이언트 초기화 및 프롬프트 정의 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("UPSTAGE_API_KEY"), base_url="https://api.upstage.ai/v1/solar")
ULTIMATE_PROMPT = "Translate the following English text to natural, formal Korean.\n\nEnglish Text:\n{text_to_translate}\n\nKorean Translation:"
print("✅ API 클라이언트 및 프롬프트가 준비되었습니다.")

# --- 2. '마스킹'을 위한 이름 및 태그 사전 ---
# 여기에 우리가 아는 모든 이름과 태그를 '비밀 코드'와 함께 정의
MASK_MAP = {
    "#Person1#": "__P1__",
    "#Person2#": "__P2__",
    "#Person3#": "__P3__",
    "#Person4#": "__P4__",
    "#Person5#": "__P5__",
    "#Person6#": "__P6__",
    "#Person7#": "__P7__",
    "Tom": "__TOM__",
    "Brian": "__BRIAN__",
    "Kate": "__KATE__",
    "Masha": "__MASHA__",
    "Hero": "__HERO__",
    "John": "__JOHN__",
    "Francis": "__FRANCIS__",
    "Steven": "__STEVEN__",
    "Tony": "__TONY__",
    "Rose": "__ROSE__",
    "Jack": "__JACK__",
    "Mike": "__MIKE__",
    "Dawson": "__DAWSON__",
    "Maya": "__MAYA__",
    "James": "__JAMES__",
    "Muriel": "__MURIEL__",
    "L.A.": "__LA__",
    # ... 필요 시 더 추가 ...
}
# 언마스킹을 위해 key-value를 뒤집은 사전도 준비
UNMASK_MAP = {v: k for k, v in MASK_MAP.items()}

def mask_text(text):
    for real, placeholder in MASK_MAP.items():
        text = text.replace(real, placeholder)
    return text

def unmask_text(text):
    for placeholder, real in UNMASK_MAP.items():
        text = text.replace(placeholder, real)
    return text

# --- 3. API 호출 및 번역 함수 ---
def api_request(messages):
    try:
        response = client.chat.completions.create(model="solar-1-mini-chat", messages=messages, temperature=0.1)
        return response.choices[0].message.content
    except Exception as e:
        print(f"🔥🔥🔥 API 호출 중 에러 발생: {e}")
        return ""

def final_translate(texts, prompt_template):
    translated = []
    for text in tqdm(texts, desc="Final Korean Translation"):
        masked_text = mask_text(str(text))
        prompt = prompt_template.format(text_to_translate=masked_text)
        messages = [{"role": "user", "content": prompt}]
        translated_masked_text = api_request(messages)
        unmasked_text = unmask_text(translated_masked_text)
        translated.append(unmasked_text)
        time.sleep(1) # Rate Limiting
    return translated

# the_ultimate_fixer.py 파일의 main 함수

# --- 4. 메인 실행 로직 (미리보기 기능 추가) ---
def main():
    SOURCE_FILE = 'submissions/SUBMISSION_FINAL.csv'
    
    print(f"--- 💡 빠른 미리보기 모드 (5개 샘플) 💡 ---")
    print(f"'{SOURCE_FILE}' 파일에서 5개의 샘플을 로드합니다...")
    try:
        df = pd.read_csv(SOURCE_FILE)
    except FileNotFoundError:
        print(f"🔥🔥🔥 에러: '{SOURCE_FILE}' 파일을 찾을 수 없습니다.")
        return

    # 전체 데이터 대신 5개 샘플만 사용
    sample_df = df.head(5)
    
    print("'마스킹' 기법을 사용하여 5개 샘플을 한국어로 번역합니다...")
    # 5개 샘플에 대해서만 번역 실행
    korean_summaries_sample = final_translate(sample_df['summary'].tolist(), ULTIMATE_PROMPT)
    
    # 임시 데이터프레임으로 결과 확인
    preview_df = pd.DataFrame({
        'fname': sample_df['fname'],
        'english_summary': sample_df['summary'],
        'korean_summary_preview': korean_summaries_sample
    })
    
    # 마지막 폴리싱 함수
    def final_polish(summary):
        text = str(summary)
        match = re.search(r'#Person', text)
        if match: text = text[match.start():]
        text = text.split("한글 번역:")[0].strip()
        return text
    
    preview_df['korean_summary_preview'] = preview_df['korean_summary_preview'].apply(final_polish)

    print("\n--- ✅ 번역 및 폴리싱 미리보기 결과 ---")
    pd.set_option('display.max_colwidth', None)
    print(preview_df[['fname', 'korean_summary_preview']])
    print("------------------------------------")
    print("\n위 결과가 만족스러우면, 이 스크립트 맨 아래에 있는 ''' 주석을 지우고 '전체 실행'을 진행하세요.")
    
    # --- 💡 전체 실행 (미리보기 결과가 만족스러우면 이 ''' 주석 세 개를 지우고 다시 실행) 💡 ---
    '''
    print("\n--- 🚀 전체 파일 번역 및 저장 시작 🚀 ---")
    all_korean_summaries = final_translate(df['summary'].tolist(), ULTIMATE_PROMPT)
    final_df = pd.DataFrame({'fname': df['fname'], 'summary': all_korean_summaries})
    final_df['summary'] = final_df['summary'].apply(final_polish)
    
    FINAL_FILE = 'submissions/SUBMISSION_FINAL_ULTIMATE_v2.csv'
    final_df.to_csv(FINAL_FILE, index=False, encoding='utf-8-sig')
    print(f"\n🎉🎉🎉 최종 파일이 생성되었습니다: '{FINAL_FILE}'")
    '''

if __name__ == "__main__":
    main()