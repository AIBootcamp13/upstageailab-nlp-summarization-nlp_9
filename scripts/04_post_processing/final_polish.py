# scripts/04_post_processing/final_polish.py
import pandas as pd
import re

print("🚀 최종 제출 파일 강화 클리닝을 시작합니다...")

SOURCE_FILE = 'submissions/SUBMISSION_FINAL_KOREAN.csv'
POLISHED_FILE = 'submissions/SUBMISSION_FINAL_POLISHED_v1.csv'

try:
    df = pd.read_csv(SOURCE_FILE)
    print(f"✅ 원본 파일 로드 완료: '{SOURCE_FILE}'")

    def polish_summary(idx, summary):
        text = str(summary)
        
        # 1. 더 강력한 프롬프트 찌꺼기 제거
        # '#Person' 또는 '#사람'이 처음 나타나는 부분을 찾는다.
        match = re.search(r'(#Person|#사람)', text)
        if match:
            # 그 부분부터 텍스트가 시작되도록 앞부분을 잘라낸다.
            text = text[match.start():]

        # 2. '#사람' 태그를 '#Person'으로 통일
        for i in range(1, 10):
            text = text.replace(f'#사람{i}#', f'#Person{i}#')
            text = text.replace(f'#사람{i}', f'#Person{i}') # '#'가 하나만 있는 경우도 처리

        # 3. 문장 잘림 가능성 확인
        # 문장의 끝이 ., ?, !, ", > 로 끝나지 않으면 경고 출력
        if not text.strip().endswith(('.', '?', '!', '"', '>')):
            print(f"  ⚠️ [경고] {idx}번 행의 문장이 잘렸을 수 있습니다: ...{text[-30:]}")

        return text

    # 'summary' 컬럼의 모든 행에 polish_summary 함수를 적용
    # 인덱스도 함께 전달하기 위해 lambda 함수 사용
    df['summary'] = [polish_summary(idx, summary) for idx, summary in df['summary'].items()]
    
    df.to_csv(POLISHED_FILE, index=False, encoding='utf-8-sig')

    print(f"\n🎉🎉🎉 최종 폴리싱 완료! 파일이 생성되었습니다: '{POLISHED_FILE}'")
    print("\n--- 폴리싱 결과 샘플 ---")
    print(df.head())

except Exception as e:
    print(f"🔥🔥🔥 에러 발생: {e}")