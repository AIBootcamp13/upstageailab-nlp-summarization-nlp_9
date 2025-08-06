import pandas as pd
import re

print("🚀 최종 제출 파일 '궁극의 폴리싱'을 시작합니다...")

# 가장 최근에 번역한, 오류가 있는 한국어 파일을 입력으로 사용
SOURCE_FILE = 'submissions/SUBMISSION_FINAL_KOREAN_v2.csv'
# 진짜 최종 제출할 파일 이름
POLISHED_FILE = 'submissions/SUBMISSION_FINAL_ULTIMATE_v3.csv'

try:
    df = pd.read_csv(SOURCE_FILE)
    print(f"✅ 원본 파일 로드 완료: '{SOURCE_FILE}'")

    def ultimate_cleaner(idx, summary):
        text = str(summary)
        
        # 1. 프롬프트 찌꺼기 제거 (가장 먼저 수행)
        match = re.search(r'(#Person|#사람)', text)
        if match:
            text = text[match.start():]

        # 2. '#사람' 태그를 '#Person'으로 통일
        for i in range(1, 10):
            text = text.replace(f'#사람{i}#', f'#Person{i}#').replace(f'#사람{i}', f'#Person{i}')

        # 3. 문장 끝의 "한글 번역:" 꼬리표 제거
        # "한글 번역:" 이라는 문자열을 기준으로 나누고, 그 앞부분만 선택
        text = text.split("한글 번역:")[0].strip()
        
        # 4. 문장 잘림 가능성 확인
        if not text.strip().endswith(('.', '?', '!', '"', '>','다','요')):
            print(f"  ⚠️ [경고] {idx}번 행의 문장이 잘렸을 수 있습니다: ...{text[-50:]}")

        return text

    df['summary'] = [ultimate_cleaner(idx, summary) for idx, summary in df['summary'].items()]
    
    df.to_csv(POLISHED_FILE, index=False, encoding='utf-8-sig')

    print(f"\n🎉🎉🎉 최종 폴리싱 완료! 파일이 생성되었습니다: '{POLISHED_FILE}'")
    print("\n--- 최종 결과 샘플 ---")
    print(df.head())

except Exception as e:
    print(f"🔥🔥🔥 에러 발생: {e}")