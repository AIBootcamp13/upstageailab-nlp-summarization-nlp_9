# prepare_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("🚀 데이터 준비를 시작합니다...")

# 1. 원본 데이터 파일 경로
source_file = 'data/processed/translation_checkpoint.csv'

# 2. 결과 저장 경로
output_dir = 'data/processed'
train_file = os.path.join(output_dir, 'train.csv')
val_file = os.path.join(output_dir, 'val.csv')

# 3. 데이터 로드
try:
    df = pd.read_csv(source_file)
    print(f"✅ 원본 데이터 로드 완료: {len(df)}개 행")
except FileNotFoundError:
    print(f"🔥🔥🔥 에러: '{source_file}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# 🔥🔥🔥 핵심 수정: 빈 요약문이 있는 행을 전부 제거! 🔥🔥🔥
original_rows = len(df)
df.dropna(subset=['english_dialogue', 'english_summary', 'english_topic'], inplace=True)
if len(df) < original_rows:
    print(f"✅ 비어있는 행 {original_rows - len(df)}개를 제거했습니다.")


# 4. 필요한 영어 컬럼만 선택
columns_to_use = ['english_dialogue', 'english_summary', 'english_topic']
df_english = df[columns_to_use].copy()

# 5. Flan-T5 모델에 최적화된 입력 형식(input_text) 생성
def create_input_text(row):
    return f"summarize: topic: {row['english_topic']} dialogue: {row['english_dialogue']}"

df_english['input_text'] = df_english.apply(create_input_text, axis=1)

# 6. 최종적으로 사용할 컬럼만 남기기
df_final = df_english[['input_text', 'english_summary']]
print("✅ 'input_text' 컬럼 생성 완료.")

# 7. 훈련 데이터와 검증 데이터로 분할 (95% 훈련, 5% 검증)
train_df, val_df = train_test_split(df_final, test_size=0.05, random_state=42)
print(f"✅ 데이터를 훈련 세트({len(train_df)}개)와 검증 세트({len(val_df)}개)로 분할 완료.")


# --- 개선점: 입력 규제를 위한 'Topic Dropout' ---
# 훈련 데이터의 10%에 해당하는 샘플들의 topic을 'unknown'으로 변경
print("✅ [개선] 훈련 데이터의 10%에 Topic Dropout을 적용합니다...")
dropout_indices = train_df.sample(frac=0.1, random_state=42).index
# 'summarize: topic: ...' 형식에서 topic 부분만 'unknown'으로 교체
train_df.loc[dropout_indices, 'input_text'] = train_df.loc[dropout_indices, 'input_text'].str.replace(r'topic: .*? dialogue:', 'topic: unknown dialogue:', regex=True)

# 8. 파일로 저장
os.makedirs(output_dir, exist_ok=True)
train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)

print(f"🎉 성공! '{train_file}'과 '{val_file}'이 생성되었습니다.")

# prepare_data.py (수정본)
#  훈련용으로 20%의 데이터만 잘라서 train_subset.csv라는 파일을 만들도록 변경
# prepare_data.py (최종 수정본)
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os

# print("🚀 데이터 준비를 시작합니다...")

# # 1. 파일 경로 설정
# source_file = 'data/processed/translation_checkpoint.csv'
# output_dir = 'data/processed'
# train_file = os.path.join(output_dir, 'train.csv')
# val_file = os.path.join(output_dir, 'val.csv')
# train_subset_file = os.path.join(output_dir, 'train_subset.csv') 

# try:
#     # 2. 원본 데이터 로드
#     df = pd.read_csv(source_file)
#     print(f"✅ 원본 데이터 로드 완료: {len(df)}개 행")

#     # 3. 빈 값이 있는 행 제거
#     original_rows = len(df)
#     df.dropna(subset=['english_dialogue', 'english_summary', 'english_topic'], inplace=True)
#     if len(df) < original_rows:
#         print(f"✅ 비어있는 행 {original_rows - len(df)}개를 제거했습니다.")

#     # --- ▼▼▼▼ 이 부분이 코드에서 빠져있었어! ▼▼▼▼ ---
    
#     # 4. 필요한 영어 컬럼만 선택
#     columns_to_use = ['english_dialogue', 'english_summary', 'english_topic']
#     df_english = df[columns_to_use].copy()

#     # 5. Flan-T5 모델에 최적화된 입력 형식(input_text) 생성
#     def create_input_text(row):
#         return f"summarize: topic: {row['english_topic']} dialogue: {row['english_dialogue']}"

#     df_english['input_text'] = df_english.apply(create_input_text, axis=1)

#     # 6. 최종적으로 사용할 컬럼만 남겨서 df_final 생성
#     df_final = df_english[['input_text', 'english_summary']]
#     print("✅ 'input_text' 컬럼 생성 완료.")
    
#     # --- ▲▲▲▲ 여기까지가 빠져있던 부분 ▲▲▲▲ ---

#     # 7. 훈련 데이터와 검증 데이터로 분할
#     train_df, val_df = train_test_split(df_final, test_size=0.05, random_state=42)
#     print(f"✅ 데이터를 훈련 세트({len(train_df)}개)와 검증 세트({len(val_df)}개)로 분할 완료.")

#     # 8. Sweep을 위한 훈련 데이터 서브셋(20%) 생성
#     train_subset_df = train_df.sample(frac=0.2, random_state=42)
#     print(f"✅ Sweep 실험용 서브셋 생성 완료: {len(train_subset_df)}개 행")

#     # 9. 모든 파일을 저장
#     os.makedirs(output_dir, exist_ok=True)
#     train_df.to_csv(train_file, index=False)
#     val_df.to_csv(val_file, index=False)
#     train_subset_df.to_csv(train_subset_file, index=False)

#     print(f"🎉 성공! 모든 파일이 생성되었습니다.")

# except FileNotFoundError:
#     print(f"🔥🔥🔥 에러: '{source_file}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
# except Exception as e:
#     print(f"🔥🔥🔥 에러 발생: {e}")