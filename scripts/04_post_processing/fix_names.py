# scripts/04_post_processing/fix_names.py
import pandas as pd
import os

print("🚀 최종 제출 파일 '이름 복원'을 시작합니다...")

# 가장 최근에 만든, 내용이 깨끗한 한국어 요약 파일
SOURCE_FILE = 'submissions/SUBMISSION_TO_SUBMIT.csv' 
# 이름까지 완벽하게 수정된 진짜 최종 제출 파일
FINAL_FILE = 'submissions/SUBMISSION_FINAL_VERSION.csv'

try:
    df = pd.read_csv(SOURCE_FILE)
    print(f"✅ 원본 파일 로드 완료: '{SOURCE_FILE}'")

    # --- 1. 우리만의 '한글-영어 이름 번역 사전' ---
    name_map = {
    # 사람
    "케이트": "Kate", "마샤": "Masha", "히어로": "Hero", "브라이언": "Brian", "스티븐": "Steven",
    "아브라함 링컨": "Abraham Lincoln", "프랜시스": "Francis", "토니": "Tony", "톰": "Tom",
    "루오자": "Ruojia", "마이크": "Mike", "프랭크": "Frank", "마야": "Maya", "제임스": "James",
    "머리얼": "Muriel", "폴리 씨": "Mr. Polly", "모니카": "Monica", "토드 부인": "Mrs. Todd",
    "빌": "Bill", "클레오": "Cleo", "마크": "Mark", "매기": "Maggie", "터너 교수": "Professor Turner",
    "버먼 교수": "Professor Berman", "사라": "Sarah", "마크 리치": "Mark Richie", "루시": "Lucy",
    "린 방": "Lin Fang", "토마스 부인": "Mrs. Thomas", "로라": "Laura", "루루": "Lulu", "빅": "Vic",
    "데이브 톰슨": "Dave Thompson", "짐": "Jim", "레아": "Leah", "네이선": "Nathan",
    "콜린스 여사": "Mrs. Collins", "류 씨": "Mr. Liu", "디크": "Dick", "제인스": "Janes",
    "애덤": "Adam", "존": "John", "레베카": "Rebecca", "메리": "Mary", "제인": "Jane",
    "수잔": "Susan", "우 씨": "Mr. Wu", "피셔 씨": "Mr. Fisher", "로스 씨": "Mr. Ross",
    "월리스": "Wallace", "브레인 로커": "Brain Locker", "톰 윌슨": "Tom Wilson",
    "캐롤": "Carol", "질": "Jill", "도널드 트럼프": "Donald Trump", "바이든": "Biden",
    "리리": "Lili", "윌슨 씨": "Mr. Wilson", "Dawson 씨": "Ms. Dawson",

}
    print(f"✅ {len(name_map)}개의 이름에 대한 번역 사전을 만들었습니다.")

    # 2. 'summary' 컬럼의 모든 행에 대해, 사전에 있는 모든 이름을 영어로 교체
    # tqdm 같은 라이브러리가 없으니, 간단한 루프로 진행
    summaries = df['summary'].tolist()
    cleaned_summaries = []
    for summary in summaries:
        text = str(summary)
        for kor_name, eng_name in name_map.items():
            text = text.replace(kor_name, eng_name)
        cleaned_summaries.append(text)
    
    df['summary'] = cleaned_summaries
    
    # 3. 최종 파일을 저장
    df.to_csv(FINAL_FILE, index=False, encoding='utf-8-sig')

    print(f"\n🎉🎉🎉 이름 복원 완료! 진짜 최종 제출 파일: '{FINAL_FILE}'")
    print("\n--- 최종 결과 샘플 ---")
    print(df.head())

except FileNotFoundError:
    print(f"🔥🔥🔥 에러: '{SOURCE_FILE}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"🔥🔥🔥 에러 발생: {e}")