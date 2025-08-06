# scripts/04_post_processing/add_spacing.py
import pandas as pd
import os

# --- 설정 ---
# 네가 번역기를 돌리고 수동 검수까지 마친 '거의 최종' 파일
SOURCE_FILE = 'submissions/SUBMISSION_FINAL_ULTIMATE_v3.csv' 
# 띄어쓰기까지 추가된 '진짜 최종' 제출 파일
FINAL_SUBMIT_FILE = 'submissions/SUBMISSION_TO_SUBMIT.csv'

print(f"'{SOURCE_FILE}' 파일의 화자 태그에 띄어쓰기를 추가합니다...")

try:
    df = pd.read_csv(SOURCE_FILE)

    def add_space_after_tags(summary):
        text = str(summary)
        # #Person1# 부터 #Person9# 까지 모든 태그를 찾아서,
        # 뒤에 공백이 없는 경우 한 칸 띄어쓰기를 추가해준다.
        for i in range(1, 10):
            text = text.replace(f'#Person{i}#', f'#Person{i}# ')
        return text

    df['summary'] = df['summary'].apply(add_space_after_tags)

    df.to_csv(FINAL_SUBMIT_FILE, index=False, encoding='utf-8-sig')

    print(f"\n🎉 띄어쓰기 수정 완료! 진짜 최종 제출 파일: '{FINAL_SUBMIT_FILE}'")
    print("\n--- 최종 결과 샘플 ---")
    print(df.head())

except FileNotFoundError:
    print(f"🔥🔥🔥 에러: '{SOURCE_FILE}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"🔥🔥🔥 에러 발생: {e}")