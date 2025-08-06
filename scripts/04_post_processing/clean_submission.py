# scripts/04_post_processing/clean_submission.py
import pandas as pd
import os

print("🚀 최종 제출 파일 클리닝을 시작합니다...")

# --- 경로 설정 ---
# 이 스크립트는 프로젝트 루트 폴더에서 실행하는 것을 기준으로 함
SOURCE_FILE = 'submissions/SUBMISSION_FINAL_KOREAN.csv'
CLEANED_FILE = 'submissions/submission_final_v1.csv'

try:
    # 1. 원본 제출 파일 로드
    df = pd.read_csv(SOURCE_FILE)
    print(f"✅ 원본 파일 로드 완료: '{SOURCE_FILE}'")

    # 2. 클리닝 함수 정의
    def clean_summary(summary):
        # API 프롬프트 지시사항이 포함된 경우, '---' 뒷부분의 실제 번역문만 남김
        if '---' in str(summary):
            # '---'를 기준으로 문장을 나누고, 마지막 부분을 선택
            cleaned_text = summary.split('---')[-1]
            # "다음 영어 요약문을 번역해 주세요:" 와 같은 불필요한 앞부분 제거
            if "Please translate the following English summary:" in cleaned_text:
                 cleaned_text = cleaned_text.split("Please translate the following English summary:")[-1]
            # 앞쪽의 불필요한 공백이나 줄바꿈 제거
            return cleaned_text.strip()
        else:
            # 정상적인 요약문은 그대로 반환
            return summary

    # 3. 'summary' 컬럼의 모든 행에 클리닝 함수 적용
    original_summaries = df['summary'].copy()
    df['summary'] = df['summary'].apply(clean_summary)
    
    # 변경된 행의 수 계산
    changed_rows = (original_summaries != df['summary']).sum()
    print(f"✅ {changed_rows}개의 행에서 불필요한 프롬프트 지시사항을 제거했습니다.")
    
    # 4. 깨끗해진 데이터를 새 파일로 저장 (버전 1)
    df.to_csv(CLEANED_FILE, index=False, encoding='utf-8-sig')

    print(f"\n🎉🎉🎉 1차 클리닝 완료! 최종 파일이 생성되었습니다: '{CLEANED_FILE}'")
    print("\n--- 클리닝 결과 샘플 ---")
    print(df.head())

except FileNotFoundError:
    print(f"🔥🔥🔥 에러: '{SOURCE_FILE}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"🔥🔥🔥 에러 발생: {e}")