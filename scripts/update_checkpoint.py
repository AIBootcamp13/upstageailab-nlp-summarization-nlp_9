# scripts/update_checkpoint.py
import pandas as pd
import os

# 업데이트할 체크포인트 파일 경로
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkpoint_path = os.path.join(project_root, 'data', 'processed', 'translation_checkpoint.csv')

print(f"'{checkpoint_path}' 파일을 업데이트합니다...")

try:
    # 체크포인트 파일을 불러온다
    df = pd.read_csv(checkpoint_path)

    # 'english_topic' 컬럼이 없는 경우에만 새로 추가한다
    if 'english_topic' not in df.columns:
        print("'english_topic' 컬럼이 없어서 새로 추가합니다.")
        df['english_topic'] = None
        
        # 변경된 내용을 다시 파일에 저장한다
        df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')
        print("✅ 파일 업데이트 성공!")
    else:
        print("✅ 이미 최신 버전의 파일입니다. 작업이 필요 없습니다.")

except FileNotFoundError:
    print(f"❌ 체크포인트 파일을 찾을 수 없습니다. 경로를 확인해주세요.")