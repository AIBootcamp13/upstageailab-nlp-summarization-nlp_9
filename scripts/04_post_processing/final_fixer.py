
import pandas as pd
import csv

print("🚀 최종 제출 파일 '궁극의 수술'을 시작합니다...")

# 네가 수동 번역하고, 여러 오류가 섞인 바로 그 파일
SOURCE_FILE = 'submissions/SUBMISSION_FINAL_ULTIMATE_v3.csv' 
# 대회에 제출할 진짜 최종 파일 이름
FINAL_SUBMISSION_FILE = 'submissions/SUBMISSION_READY.csv'

try:
    fnames = []
    summaries = []

    print(f"✅ 원본 파일을 텍스트 모드로 로드합니다: '{SOURCE_FILE}'")
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        # 헤더(첫 줄)는 건너뛰기
        next(f)
        for line in f:
            # 1. '첫 번째 쉼표'를 기준으로만 데이터를 나눔 (가장 중요!)
            #    summary 안에 쉼표가 있어도 절대 깨지지 않음
            try:
                fname, summary = line.strip().split(',', 1)
            except ValueError:
                # 쉼표가 없는 빈 줄 등은 건너뛰기
                print(f"  ⚠️ [경고] 파싱 오류가 있는 행을 건너뜁니다: {line.strip()}")
                continue
            
            # 2. summary 앞뒤의 모든 종류의 따옴표와 공백을 제거
            summary = summary.strip().strip('"“”')

            # 3. 띄어쓰기 추가
            for i in range(1, 10):
                summary = summary.replace(f'#Person{i}#', f'#Person{i}# ')

            fnames.append(fname)
            summaries.append(summary)

    print("✅ 파일 파싱 및 클리닝 완료.")
    
    # 4. 깨끗해진 데이터로 새로운 데이터프레임 생성
    df = pd.DataFrame({
        'fname': fnames,
        'summary': summaries
    })

    # 5. 대회 제출 형식에 맞게 최종 저장 (quoting 옵션 수정)
    df.to_csv(
        FINAL_SUBMISSION_FILE,
        index=True,
        index_label='',
        quoting=csv.QUOTE_MINIMAL, # <-- QUOTE_NONE에서 QUOTE_MINIMAL로 수정
        encoding='utf-8-sig'
    )

    print(f"\n🎉🎉🎉 모든 문제가 해결된 최종 파일이 생성되었습니다: '{FINAL_SUBMISSION_FILE}'")
    # ... (이하 동일)
    print(f"\n🎉🎉🎉 모든 문제가 해결된 최종 파일이 생성되었습니다: '{FINAL_SUBMISSION_FILE}'")
    print("\n--- 최종 결과 샘플 ---")
    print(df.head())

except FileNotFoundError:
    print(f"🔥🔥🔥 에러: '{SOURCE_FILE}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"🔥🔥🔥 에러 발생: {e}")