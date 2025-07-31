# 자연어 힌트, 계층적 샘플링, 정제된 데이터셋 생성 스크립트

import pandas as pd
from sklearn.model_selection import train_test_split
import os


def clean_text(text):
    """텍스트 전처리: 공백 제거, 줄바꿈 제거, 소문자화 등"""
    if pd.isna(text):
        return ""
    text = text.strip().replace("\n", " ").replace("\r", " ")
    return text.lower()


def filter_and_clean_dataframe(df):
    """필터링 + 정제"""
    # 체크포인트 파일에 번역 실패(TRANSLATION_FAILED)나 결측치(NaN)가 있을 경우를 대비해 필터링
    df = df.dropna(subset=['english_dialogue', 'english_summary', 'english_topic'])
    df = df[~df['english_dialogue'].str.contains("TRANSLATION_FAILED", na=False)]
    df = df[~df['english_summary'].str.contains("TRANSLATION_FAILED", na=False)]
    df = df[~df['english_topic'].str.contains("TRANSLATION_FAILED", na=False)]

    # 텍스트 클리닝 적용
    df['english_dialogue'] = df['english_dialogue'].apply(clean_text)
    df['english_summary'] = df['english_summary'].apply(clean_text)
    df['english_topic'] = df['english_topic'].str.lower().str.strip()
    return df

def main():
    """
    [V7] Stratify 옵션을 제거하여 데이터 분할 에러를 최종 해결.
    이제 모든 데이터를 활용하여 랜덤 분할을 수행.
    """
    print("🚀 [V7] 최종 데이터셋 준비를 시작합니다...")

    # --- 1. 데이터 로드 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(project_root, 'data', 'processed', 'translation_checkpoint.csv')

    print(f"번역된 체크포인트 파일을 불러옵니다: {checkpoint_path}")
    df = pd.read_csv(checkpoint_path)

    # --- 2. 정제 및 전처리 ---
    df = filter_and_clean_dataframe(df)
    print(f"정제 및 전처리 후 데이터 크기: {len(df)}개")

    # --- 3. 자연어 힌트 input_text 생성 ---
    df['input_text'] = "summarize: topic: " + df['english_topic'] + ". dialogue: " + df['english_dialogue']
    
    # Stratify를 사용하지 않으므로, 더 이상 english_topic 컬럼은 필요 없음
    final_df = df[['input_text', 'english_summary']]

    # --- 4. Random split (랜덤 분할) ---
    print("데이터셋을 '랜덤' 방식으로 분할합니다 (Stratify 옵션 제거).")
    train_df, val_df = train_test_split(
        final_df,
        test_size=0.1,
        random_state=42, # 결과 재현을 위해 random_state는 유지
        shuffle=True,    # 섞어주는 것도 유지
        # stratify 옵션만 제거
    )

    # --- 5. 저장 ---
    train_output_path = os.path.join(project_root, 'data', 'processed', 'train.csv')
    val_output_path = os.path.join(project_root, 'data', 'processed', 'val.csv')

    train_df.to_csv(train_output_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_output_path, index=False, encoding='utf-8-sig')

    print("\n✅ 훈련/검증 데이터셋 저장 완료!")
    print(f"  - 훈련셋 경로: {train_output_path} ({len(train_df)}개)")
    print(f"  - 검증셋 경로: {val_output_path} ({len(val_df)}개)")
if __name__ == "__main__":
    main()