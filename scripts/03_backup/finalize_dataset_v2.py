''' Flan-T5 같은 최신 모델은 '자연어'를 이해하는 능력이 매우 뛰어나다. 
따라서 <special_token> 같은 인공적인 기호를 쓰는 것보다, 자연스러운 문장 형태의 힌트를 주는 것이 훨씬 더 효과적이다.'''

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    """
    [V2] 체크포인트 파일로부터, Topic을 '자연어 힌트'로 사용하여
    최종 훈련/검증 데이터셋을 생성한다.
    """
    print("🚀 [V2] 최종 데이터셋 준비를 시작합니다...")

    # --- 1. 데이터 로드 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_path = os.path.join(project_root, 'data', 'processed', 'translation_checkpoint.csv')

    print(f"'{checkpoint_path}'에서 체크포인트 데이터를 불러옵니다...")
    df = pd.read_csv(checkpoint_path)

    # --- 2. 데이터 정제 및 전처리 ---
    df = df.dropna(subset=['english_dialogue', 'english_summary', 'english_topic'])
    df = df[~df['english_dialogue'].str.contains("TRANSLATION_FAILED", na=False)]
    df = df[~df['english_summary'].str.contains("TRANSLATION_FAILED", na=False)]
    df = df[~df['english_topic'].str.contains("TRANSLATION_FAILED", na=False)]
    
    print("텍스트 데이터를 소문자로 변환합니다...")
    df['english_dialogue'] = df['english_dialogue'].str.lower()
    df['english_summary'] = df['english_summary'].str.lower()
    df['english_topic'] = df['english_topic'].str.lower()
    
    # ▼▼▼▼▼ 핵심: '자연어 힌트' 포맷으로 입력 텍스트 생성 ▼▼▼▼▼
    df['input_text'] = "topic: " + df['english_topic'] + ". dialogue: " + df['english_dialogue']
    print("✅ '자연어 힌트'가 포함된 입력 텍스트 생성 완료!")
    
    # 최종적으로 모델 훈련에 필요한 컬럼만 선택
    final_df = df[['input_text', 'english_summary']]

    # --- 3. 데이터셋 분할 및 저장 ---
    print("데이터셋을 훈련용과 검증용으로 분할합니다...")
    train_df, val_df = train_test_split(final_df, test_size=0.1, random_state=42, shuffle=True)

    train_output_path = os.path.join(project_root, 'data', 'processed', 'train.csv')
    val_output_path = os.path.join(project_root, 'data', 'processed', 'val.csv')

    train_df.to_csv(train_output_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_output_path, index=False, encoding='utf-8-sig')

    print("\n✅ 훈련/검증 데이터셋 저장 완료!")
    print(f"  - 훈련셋 경로: {train_output_path} ({len(train_df)}개)")
    print(f"  - 검증셋 경로: {val_output_path} ({len(val_df)}개)")

if __name__ == "__main__":
    main()