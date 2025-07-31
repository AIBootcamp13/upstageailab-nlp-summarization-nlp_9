import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    """
    [Simple Ver] 두 개의 번역 파일을 합치고, 전처리하고,
    최종 훈련/검증 데이터셋으로 분할하여 저장하는 스크립트.
    Topic 컬럼은 일단 제외한다.
    """
    print("🚀 [Simple Ver] 최종 데이터셋 준비를 시작합니다...")

    # --- 1. 데이터 로드 및 병합 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    restored_dialogue_path = os.path.join(project_root, 'data', 'processed', 'train_restored_english.csv')
    final_summary_path = os.path.join(project_root, 'data', 'processed', 'train_final_english.csv')

    print("중간 데이터 파일들을 불러옵니다...")
    dialogue_df = pd.read_csv(restored_dialogue_path)
    summary_df = pd.read_csv(final_summary_path)

    df = pd.DataFrame({
        'english_dialogue': dialogue_df['english_summary'], # dialogue 번역본
        'english_summary': summary_df['english_summary'],  # summary 번역본
    })
    
    # --- 2. 데이터 정제 및 전처리 ---
    df = df.dropna()
    df = df[~df['english_dialogue'].str.contains("TRANSLATION_FAILED", na=False)]
    df = df[~df['english_summary'].str.contains("TRANSLATION_FAILED", na=False)]
    
    print("텍스트 데이터를 소문자로 변환합니다...")
    df['english_dialogue'] = df['english_dialogue'].str.lower()
    df['english_summary'] = df['english_summary'].str.lower()
    print(f"정제 및 전처리 후 데이터 크기: {len(df)}개")

    # --- 3. 데이터셋 분할 및 저장 ---
    print("데이터셋을 훈련용과 검증용으로 분할합니다...")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, shuffle=True)

    train_output_path = os.path.join(project_root, 'data', 'processed', 'train.csv')
    val_output_path = os.path.join(project_root, 'data', 'processed', 'val.csv')

    train_df.to_csv(train_output_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_output_path, index=False, encoding='utf-8-sig')

    print("\n✅ 훈련/검증 데이터셋 저장 완료!")
    print(f"  - 훈련셋 경로: {train_output_path} ({len(train_df)}개)")
    print(f"  - 검증셋 경로: {val_output_path} ({len(val_df)}개)")

if __name__ == "__main__":
    main()

    