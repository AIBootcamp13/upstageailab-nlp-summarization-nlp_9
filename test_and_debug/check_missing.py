# check_missing.py
import pandas as pd

# 원본 데이터 파일 경로
source_file = 'data/processed/translation_checkpoint.csv'

print(f"'{source_file}' 파일의 결측치를 확인합니다...")

try:
    df = pd.read_csv(source_file)

    # 'english_summary' 컬럼에 있는 결측치(NaN, 비어있는 값)의 총 개수를 센다.
    missing_count = df['english_summary'].isnull().sum()

    if missing_count > 0:
        print(f"\n🔥🔥🔥 'english_summary' 컬럼에서 총 {missing_count}개의 결측치를 발견했습니다.")
        
        print("\n--- 결측치가 포함된 행 샘플 (상위 5개) ---")
        # 'english_summary' 컬럼이 비어있는 행들만 필터링해서 보여주기
        missing_rows = df[df['english_summary'].isnull()]
        print(missing_rows.head())

        print("\n(위 행들은 'english_summary'가 비어있어, 이전 단계에서 훈련 에러를 유발했을 가능성이 높습니다.)")

    else:
        print("\n✅ 'english_summary' 컬럼에 결측치가 없습니다. 모든 데이터가 정상입니다!")

except FileNotFoundError:
    print(f"에러: '{source_file}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except KeyError:
    print(f"에러: 파일에 'english_summary' 컬럼이 없습니다. 컬럼 이름을 확인해주세요.")