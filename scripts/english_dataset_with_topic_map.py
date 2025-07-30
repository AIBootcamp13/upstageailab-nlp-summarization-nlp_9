# scripts/english_dataset_with_topic_map.py
# topic_map은, 전체 데이터(12,000개)가 아니라 테스트용으로 뽑았던 작은 샘플(700개)에 들어있던 토픽들만으로 만든 거라서
# 전체 데이터셋인 12000개에 있는 토픽들에 대해서는 topic_map이 없을 수 있음
# 그래서 다시 solar api 를 통해서 topic_map을 생성하는 파일 생성해야겠다.

# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os

# def main():
#     """
#     번역이 완료된 체크포인트 파일로부터 최종 훈련/검증 데이터셋을 생성한다.
#     """
#     print("🚀 체크포인트로부터 최종 데이터셋 생성을 시작합니다...")

#     # --- 1. 체크포인트 파일 로드 ---
#     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     checkpoint_path = os.path.join(project_root, 'data', 'processed', 'translation_checkpoint.csv')

#     print(f"'{checkpoint_path}'에서 체크포인트 데이터를 불러옵니다...")
#     try:
#         df = pd.read_csv(checkpoint_path)
#     except FileNotFoundError:
#         print(f"❌ 체크포인트 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
#         return

#     # --- 2. 데이터 정제 및 전처리 ---
#     df = df.dropna(subset=['english_dialogue', 'english_summary', 'topic'])
#     df = df[~df['english_dialogue'].str.contains("TRANSLATION_FAILED", na=False)]
#     df = df[~df['english_summary'].str.contains("TRANSLATION_FAILED", na=False)]
    
#     print("텍스트 데이터를 소문자로 변환합니다...")
#     df['english_dialogue'] = df['english_dialogue'].str.lower()
#     df['english_summary'] = df['english_summary'].str.lower()
    
#     print("Topic 컬럼을 영어로 변환하고 스페셜 토큰을 생성합니다...")
#     # 여기에 노트북에서 완성했던 전체 topic_translation_map을 붙여넣으면 가장 좋아.

#     print("Topic 컬럼을 영어로 변환하고 스페셜 토큰을 생성합니다...")
    
#     # ▼▼▼▼▼ 여기부터 아래 딕셔너리로 교체 ▼▼▼▼▼
#     topic_map = {
#         '건강검진': 'health checkup', '백신 접종': 'vaccination', '열쇠 분실': 'lost key', '여자친구와의 결혼': 'marriage with girlfriend', 
#         '춤 제안': 'dance proposal', '쇼핑': 'shopping', '전화 통화': 'phone call', '면접': 'job interview', 
#         '음식 주문': 'food ordering', '인터뷰': 'interview', '생일 축하': 'birthday celebration', '택시 요금 설명': 'taxi fare explanation', 
#         '취업 지원': 'job application', '컴퓨터 패키지 구매': 'computer package purchase', '비자 발급 준비': 'visa preparation', 
#         '가게 심부름': 'store errand', '사진 인화': 'photo printing', '성희롱 신고': 'reporting sexual harassment', 
#         '가족과 은퇴 생활': 'family and retirement life', '음악 취향': 'music taste', '의도 파악의 혼란': 'confusion about intentions', 
#         '발렌타인데이': 'valentines day', '삼둥이와 육아': 'triplets and parenting', '아이스크림 선택': 'choosing ice cream', 
#         '공동 작업 제안': 'collaboration proposal', '차량 내 옷 갈아입기 논쟁': 'car changing clothes argument', '공항 체크인 과정': 'airport check-in process', 
#         '이사 문의': 'moving inquiry', '연인 이별': 'breakup', '영화 감상': 'watching a movie', '바 방문': 'visiting a bar', 
#         '버스 이용': 'using the bus', '직장 퇴사': 'resigning from work', '집안일 분담': 'sharing chores', 
#         '가죽 재킷과 드레스 구매': 'buying a leather jacket and dress', '로스앤젤레스의 기후와 대기 오염': 'climate and air pollution in Los Angeles', 
#         '할리우드 연구 자료 검색': 'searching hollywood research materials', '파리 여행': 'trip to Paris', '영화 선택과 대체 계획': 'movie choice and alternative plan', 
#         '일본 프로그램 준비': 'preparing for a Japan program', '시골 캠핑과 주말': 'country camping and weekend', '여자친구 관련 갈등': 'conflict with girlfriend', 
#         '저녁 식사 중의 사고': 'accident during dinner', '드레스 쇼핑': 'dress shopping', '약속 전달': 'relaying a message', 
#         '의사와의 전화 연결': 'phone call with a doctor', '지하철 책 공유': 'subway book sharing', '생일 선물 갈등': 'birthday gift conflict', 
#         'TV 시청 갈등': 'tv watching conflict', '회의 참석 준비': 'preparing for a meeting', '런던에서의 숙소 선택': 'choosing accommodation in London', 
#         '길 안내': 'giving directions', '대학교 활동 경험': 'university activity experience', '심문': 'interrogation', 
#         '면접 인터뷰': 'job interview', '신용카드 수령': 'receiving a credit card', '책 대여': 'renting a book', 
#         '결제 조건 협상': 'negotiating payment terms', '은행 계좌 개설': 'opening a bank account', '영화 연기 토론': 'discussing film acting', 
#         '여름 휴가 선호': 'summer vacation preference', '눈보라 속 차량 고립': 'car stranded in a blizzard', '상 수상 축하': 'award celebration', 
#         '파티': 'party', '미국 유학 준비': 'preparing to study in the US', '강의 선택 고민': 'worrying about course selection', 
#         "영화 '라이온 킹' 리뷰": 'review of "The Lion King"', '나이트클럽에서 춤추기': 'dancing at a nightclub', '이사 계획': 'moving plan', 
#         '에어컨 수리 요청': 'requesting air conditioner repair', '응급 상황 대처': 'handling an emergency', '티켓 구매': 'buying tickets', 
#         '진공청소기 고장 문의': 'vacuum cleaner malfunction inquiry', '물품 배송 협상': 'negotiating item delivery', '객실 예약': 'room reservation', 
#         '전화 연결 문제': 'phone connection problem', '메시지 전달': 'delivering a message', '임대 계약 협상': 'negotiating a lease', 
#         '어머니를 돌보는 아들': 'son taking care of mother', '데이트 계획': 'date plan', '신생아 수면 문제': 'newborn sleep problems', 
#         '책 찾기': 'finding a book', '3차 흡연': 'thirdhand smoke', '자판기 문제 해결': 'vending machine problem solving', 
#         '취업 면접': 'job interview', '학생의 무례한 행동': "student's rude behavior", '택시 탑승': 'taking a taxi', 
#         '숙제 지시': 'homework instruction', '망고 할인': 'mango discount', '부모 자식 갈등': 'parent-child conflict', 
#         '작별 인사 전화': 'farewell phone call', '야생 동물 보호 구역 방문 논의': 'discussing a visit to a wildlife sanctuary', 
#         '언니 귀국 및 파티 계획': "sister's return and party plan", '호텔 예약': 'hotel reservation', '여행 경험': 'travel experience', 
#         '시계 수리': 'watch repair', '연말 저녁 식사 계획': 'year-end dinner plan', '유럽 여행과 가족 방문': 'europe trip and family visit', 
#         '독서 습관 설문조사': 'reading habits survey', '집 임대': 'renting a house', '꽃다발 구매': 'buying a bouquet', 
#         '케이블 서비스 문제': 'cable service problem', 'MP3 플레이어 구매': 'buying an MP3 player', '계정 인수': 'account takeover', 
#         '연인과의 이별': 'breakup with a partner', '크리스마스 인사와 관습': 'christmas greetings and customs', '춤': 'dance', 
#         '골프 경기 중 모래 함정': 'sand trap in a golf game', '애플 제품': 'apple products', '주말 하이킹 계획': 'weekend hiking plan', 
#         '금연구역': 'no-smoking area', '자녀 비난': 'criticizing a child', '수면 문제': 'sleep problems', '커피 취향': 'coffee preference', 
#         '통화 연결 문제': 'call connection issues', '새로운 이웃과의 만남': 'meeting new neighbors', '디저트': 'dessert', 
#         '수표와 여행자 수표 교환': 'exchanging checks and traveler_s checks', '호텔 방 예약': 'hotel room reservation', '항공편 예약': 'flight reservation', 
#         '분실물 찾기': 'finding lost items', '구매 요청 승인': 'approving a purchase request', '신용카드 신청': 'credit card application', 
#         '택시 예약': 'taxi reservation', '가족 소개': 'family introduction', '학교 웹사이트 작업 보조': 'assisting with school website work', 
#         '식사 선택': 'meal selection', '식당에서의 불만 해결': 'resolving complaints at a restaurant', '하와이 여행': 'trip to Hawaii', 
#         '위험한 경험과 사고': 'dangerous experiences and accidents', '자녀 학교 생활': "child's school life", '감기': 'a cold', 
#         '겨울 휴가 활동': 'winter vacation activities', '인터넷을 통한 연락': 'contacting via the internet', '임신과 패션': 'pregnancy and fashion', 
#         '수영장 개장 및 여름 활동': 'pool opening and summer activities', '차 구매 결정': 'car purchase decision', '연애 고민': 'relationship worries', 
#         '성능 비교': 'performance comparison', '비행기 지연': 'flight delay', '공격 사건 조사': 'assault incident investigation', 
#         '영상 촬영에 대한 불만': 'complaint about filming', '연구 과제': 'research assignment', '물류 배송': 'logistics delivery', 
#         '의사의 진찰': "doctor's examination", '이직 고려': 'considering a job change', '차 렌트': 'renting a car', '경치 감상': 'enjoying the scenery', 
#         '바다 여행': 'trip to the sea', '사무실 흡연 문제': 'office smoking issue', '식당 계산': 'paying at a restaurant', 
#         '취업 면접 준비': 'job interview preparation', '업무 지연': 'work delay', '검사 결과': 'test results', '간호와 돌봄': 'nursing and care', 
#         '출산': 'childbirth', '음식 주문 지연': 'food order delay', '집들이 파티': 'housewarming party', '외모 묘사': 'describing appearance', 
#         "Janet의 매력과 재산": "janet's charm and wealth", '자연 속과 도시 생활 비교': 'comparing nature and city life', '생일 파티': 'birthday party', 
#         '도서관 안내 요청': 'requesting library assistance', '미술 도구 구매': 'buying art supplies', '공사 진행': 'construction progress', 
#         '체크아웃': 'checking out', '작별 전화': 'farewell call', '승인서 공동 서명': 'co-signing an approval form', 
#         '실험에 대한 논의': 'discussion about an experiment', '하이킹 초대': 'hiking invitation', '저녁 준비하기': 'preparing dinner', 
#         '영화 선택': 'movie selection', '미래 예측': 'predicting the future', '미국 방문 준비': 'preparing for a US visit', 
#         '신발 구매': 'buying shoes', '아침 준비': 'preparing breakfast', '콘서트 초대': 'concert invitation', 
#         '뉴욕행 비행편 정보 문의': 'inquiring about flights to New York', '크리스마스 선물 준비': 'preparing christmas gifts', '여름 방학 계획': 'summer vacation plans', 
#         '직장 내 여성 차별': 'gender discrimination at work', '한국어 수업 등록 논의': 'discussing korean class registration', '선물 구매': 'buying a gift', 
#         '화이트데이 기념 계획': 'white day celebration plans', '텔레마케팅': 'telemarketing', '꽃 주문': 'ordering flowers', '논문 작성': 'writing a thesis', 
#         '외국어 학습': 'learning a foreign language', '우산의 다양한 활용': 'various uses of an umbrella', '긴급 전화': 'emergency call', 
#         '겨울 코트 구매': 'buying a winter coat', '산악 여행 계획': 'mountain travel plans', '좋아하는 계절': 'favorite season', 
#         '외환거래': 'foreign exchange trading', '학업 및 취미 질문': 'questions about studies and hobbies', '날씨 대화': 'weather conversation', 
#         '뉴욕 임대료': 'New York rent', '건강 문제': 'health issues', 'NBA 경기 관람': 'watching an NBA game', '체중 감량': 'weight loss', 
#         '노모 걱정': 'worrying about an elderly mother', '파티 초대': 'party invitation', '파티에서의 만남': 'meeting at a party', 
#         '도서관 이용 안내': 'library usage guide', '집 꾸미기': 'decorating a house', '안전금고 대여': 'renting a safe deposit box', 
#         '슈퍼마켓 쿠폰 사용': 'using supermarket coupons', '가방 강탈': 'bag snatching', '집 인테리어 및 주방 리모델링': 'home interior and kitchen remodeling', 
#         '체중 감량 방법': 'weight loss methods', '휴대폰 요금제 추천': 'recommending a mobile plan', '대학생활과 취미': 'college life and hobbies', 
#         '비서 직무 인터뷰': 'secretary job interview', '가족 및 친구 모임': 'family and friend gathering', '면접 약속': 'interview appointment', 
#         '일에 대한 적응': 'adapting to work', '도서 대출과 연체료': 'book loans and late fees', '취미와 자기 관리': 'hobbies and self-care', 
#         '티켓 항소': 'appealing a ticket', '프로젝트 회의 일정': 'project meeting schedule', '길 묻기': 'asking for directions', 
#         '잃어버린 목걸이': 'lost necklace', '철도 국유화 논쟁': 'railway nationalization debate', '치마 구매': 'buying a skirt', 
#         '리셉셔니스트 면접자 평가': 'evaluating a receptionist candidate', '배관 수리': 'plumbing repair', '정보 유출': 'information leakage', 
#         '학교 선택': 'school choice', '질투로 인한 거짓말': 'lying due to jealousy', '오토바이 도난 사건': 'motorcycle theft incident', 
#         '시험 합격': 'passing an exam', '호텔 체크인': 'hotel check-in', '해변 여행 계획': 'beach trip plan', 
#         '주말 테니스 계획': 'weekend tennis plan', '컴퓨터 수리 요청': 'requesting computer repair', '유명 셰프 인터뷰': 'interview with a famous chef', 
#         '도서관 책 검색': 'searching for a library book', '전공 선택': 'choosing a major', '고급 초콜릿 제안': 'offering fine chocolate', 
#         '면접 준비': 'interview preparation', '휴가 요청': 'vacation request', '집 구매 제안': 'offer to buy a house', 
#         '토네이도 경보': 'tornado warning', '주말 영화 감상': 'watching a movie on the weekend', '이사 걱정': 'moving worries', 
#         '도서 반납 및 연장': 'book return and extension', '할로윈 트릭 오어 트릿': 'halloween trick or treat', '귀찮은 사람 불만': 'complaint about an annoying person', 
#         '금연 제안': 'suggesting to quit smoking', '돈과 행복': 'money and happiness', '언어 교환': 'language exchange', '장학금 신청': 'scholarship application', 
#         '경력 및 교육 인터뷰': 'career and education interview', '이사회 준비': 'board meeting preparation', '주식 투자와 손실': 'stock investment and loss', 
#         '연료 주입': 'refueling', '영어 연습': 'english practice', '약속': 'appointment', '주름과 잡티 방지 제품 추천': 'recommending anti-wrinkle products', 
#         '전문직 가족 배경과 컴퓨터 경력': 'professional family background and computer career', '가족 정보': 'family information', 
#         '장기 목표에 대한 대화': 'conversation about long-term goals', '선급금 대출 요청': 'requesting an advance loan', '출산 소식': 'news of childbirth', 
#         '직장 사임': 'resigning from work', '영어 과제 작문': 'writing an english assignment', '체크아웃 시간 문의': 'inquiring about check-out time', 
#         '축구 경기 관람': 'watching a soccer game', '응급 상황에서의 출산': 'childbirth in an emergency', '기차 연착': 'train delay', 
#         '장애인 편의 시설': 'facilities for the disabled', '잘못된 장소': 'wrong place', 'Bruce 응원': 'cheering for Bruce', 
#         '실종 신고': 'reporting a missing person', '은퇴 후 계획': 'post-retirement plans', '총지배인 승진 소문': 'rumor of general manager promotion', 
#         '시골 생활과 농장': 'country life and farm', '술집에서 대화 시작하기': 'starting a conversation at a bar', '가게 경쟁과 오해': 'store competition and misunderstanding', 
#         '운동 습관과 식생활': 'exercise habits and diet', '공항 이동 준비': 'preparing for airport transfer', '가사 일 처리': 'handling household chores', 
#         '상영 정보 제공 및 특수 지원 예약': 'providing showtimes and booking special assistance', '전화벨 소리 논의': 'discussing ringer sounds', 
#         '목도리 찾기': 'looking for a scarf', '코 푸는 소리로 인한 불만': 'complaint about nose blowing sound', '공항 보안 검사': 'airport security check', 
#         '가구 구매': 'furniture purchase', '게임': 'game', '청소 장치 소개': 'introducing a cleaning device', '고객 유치 전략': 'customer acquisition strategy', 
#         '주말 계획': 'weekend plans', '향수병과 외로움': 'homesickness and loneliness', '옷 오염 사고': 'clothing stain accident', 
#         '영어 수업 일정 조율': 'coordinating english class schedule', '노인의 결혼과 가십': 'elderly marriage and gossip', '연애 이야기': 'love story', 
#         '지붕 누수': 'leaky roof', '아파트 관람 예약': 'apartment viewing appointment', '회의 일정 잡기': 'scheduling a meeting', '지갑 분실': 'lost wallet', 
#         '전화 걸기 방법': 'how to make a phone call', '복지 혜택': 'welfare benefits', '낚시 장소 추천': 'recommending a fishing spot', 
#         '브로치 구매': 'buying a brooch', '해외 출장 및 연수 프로그램': 'overseas business trip and training program', '학교 선택 고민': 'worrying about school choice', 
#         '호텔 체크아웃': 'hotel checkout', '수업 지각': 'being late for class', '취미와 글쓰기': 'hobbies and writing', 
#         '작가 롤링과의 인터뷰': 'interview with author Rowling', '세탁 의뢰': 'laundry request', '비행기 티켓 구매 문제': 'problem with plane ticket purchase', 
#         '의류 가격 협상': 'clothing price negotiation', '처벌 경험': 'punishment experience', '침실 세트 구매': 'buying a bedroom set', 
#         '온라인 감정': 'online emotions', '지각': 'tardiness', '유럽 여행 계획': 'europe trip plan', '직업 인터뷰': 'job interview', 
#         '직장 스트레스와 격려': 'work stress and encouragement', '언어 학습': 'language learning', '수면 문제와 해결 방안': 'sleep problems and solutions', 
#         '비행기 예약': 'flight booking', '대출 상담': 'loan consultation', '밴드 참여 권유': 'invitation to join a band', '약혼 반지 구입': 'buying an engagement ring', 
#         '무도회 준비': 'prom preparation', '세계 여러 나라에서의 공부 경험': 'experience studying in various countries', '일정 조율': 'schedule coordination', 
#         'ATM 사용 문의': 'ATM usage inquiry', '청바지 구매': 'buying jeans', '미용 서비스 가격 및 선택': 'beauty service price and selection', 
#         '대출 문의': 'loan inquiry', '아기 건강 상담': 'baby health consultation', '직장 내 안전 문제': 'workplace safety issues', 
#         '시어머니 불만': 'complaint about mother-in-law', '붐비는 버스': 'crowded bus', '전시회 투어': 'exhibition tour', '외화 계좌 개설': 'opening a foreign currency account', 
#         '영화 이야기': 'movie talk', '비행기에서의 청혼': 'proposal on an airplane', '학습 격려': 'study encouragement', '용선 축제': 'dragon boat festival', 
#         '교통사고 책임': 'traffic accident liability', '파티 후 정리': 'cleaning up after a party', '헤어스타일 불만': 'hair style complaint', 
#         '이혼과 자녀 고민': 'divorce and child concerns', '요리와 연기 경력': 'cooking and acting career', '구직 제안': 'job offer', '음악 선호': 'music preference', 
#         '이발': 'haircut', '책과 잡지에 대한 관심': 'interest in books and magazines', '파티 예약 취소': 'party reservation cancellation', 
#         '도움 요청과 지원': 'request for help and support', '사진 사이즈 변경 요청': 'photo size change request', 'Ben에 대한 논의': 'discussion about Ben', 
#         '헤어스타일링': 'hairstyling', '방의 누수': "room's water leak", '베이징 여행 예약': 'beijing trip reservation', '세탁물 서비스 요청': 'laundry service request', 
#         '보고서 복사 요청': 'report copy request', '레스토랑 대화': 'restaurant conversation', '날씨 안내': 'weather forecast', 
#         '홈스테이 선정과 생활': 'homestay selection and life', '회계사 취업 면접': 'accountant job interview', '헤어스타일 칭찬': 'complimenting a hairstyle', 
#         '약혼 소식': 'engagement news', '차량 예약 문제': 'car reservation problem', '정장 구매': 'buying a suit', '경기 패배': 'losing a game', 
#         '해외 유학 계획': 'study abroad plan', '스케이트보드 가게 창업': 'starting a skateboard shop', '여행 계획 변경': 'changing travel plans', 
#         '주택 임대': 'housing rental', '사진 촬영': 'photography', '생일 선물 쇼핑': 'birthday gift shopping', 'TV 시청 조정': 'adjusting tv watching', 
#         '채소 먹기': 'eating vegetables', '가격 협상': 'price negotiation', '식료품 구매': 'grocery shopping', '월급 미지급': 'unpaid salary', 
#         '저녁 초대': 'dinner invitation', '첫 춤 초대': 'first dance invitation', '잘못된 주문과 음식 문제': 'wrong order and food problem', 
#         '생일 파티 초대': 'birthday party invitation', '영화 취향': 'movie preference', '스테레오 구경': 'looking at a stereo', 
#         '합창단 신입': 'new choir member', '깜짝 선물': 'surprise gift', '스웨터 구매': 'buying a sweater', '국제 송금': 'international money transfer', 
#         '발령지 공유와 광둥어 학습': 'sharing new post and learning cantonese', '투자 결정': 'investment decision', '휴대폰 구매 상담': 'cell phone purchase consultation', 
#         '취미': 'hobby', '바쁜 업무': 'busy work', '시계 구매': 'buying a watch', '상사와의 대화': 'conversation with a boss', 
#         '식당 평가': 'restaurant review', '문제 해결 요구': 'demand for problem solving', '불면증과 스트레스': 'insomnia and stress', 
#         '편의 시설 요금 문제': 'amenity fee issue', '이력서 작성': 'writing a resume', '적합한 후보 논의': 'discussing suitable candidates', 
#         '객실 요금 문의': 'room rate inquiry', '전화 메시지 전달': 'relaying a phone message', '좋은 소식 신문': 'good news newspaper', 
#         '주거침입 의혹': 'suspicion of trespassing', '웨딩드레스와 체중 감량': 'wedding dress and weight loss', '출근 통보': 'notice to report to work', 
#         '뉴 사이언티스트 읽기': 'reading new scientist', '저녁 시간 취미': 'evening hobby', '부녀 데이트 협상': 'father-daughter date negotiation', 
#         '매출 하락과 품질 문제': 'sales decline and quality issues', '화성 탐사': 'mars exploration', '영화 예매 취소': 'movie ticket cancellation', 
#         '해고 소문': 'layoff rumor', '격려와 존경': 'encouragement and respect', '중국의 아름다움 기준과 사랑': "china's beauty standards and love", 
#         '호텔 예약 확인': 'hotel reservation confirmation', '도서 인터뷰': 'book interview', '밴드 조직': 'forming a band', '직장 내 병결': 'sick leave at work', 
#         '건강과 환경 의식': 'health and environmental awareness', '스마트폰 구매': 'smartphone purchase', '불면': 'insomnia', '저녁 준비': 'dinner preparation', 
#         '수출 신용장 수령': 'receiving an export letter of credit', '꽃 구매': 'flower purchase', '낙태 논쟁': 'abortion debate', '자리 문의': 'seat inquiry', 
#         '데이트 이야기': 'date story', '과일 구매': 'fruit purchase', '독립기념일 캠핑 계획': 'independence day camping plan', '학업과 시험': 'studies and exams', 
#         '뮤지컬 초대': 'musical invitation', '옷 쇼핑': 'clothes shopping', '기금 모금 준비': 'fundraising preparation', '헬스장 방문': 'gym visit', 
#         '춤 초대': 'dance invitation', '계산서 검토': 'bill review', '복싱 관람': 'watching boxing', '섣부른 판단에 대한 논쟁': 'debate over hasty judgment', 
#         '아침식사 제안': 'breakfast suggestion', '작업 마감 지원': 'deadline assistance', '독감 진단 및 치료': 'flu diagnosis and treatment', 
#         '가치관 변화': 'change in values', '운영 견학': 'operations tour', '잘못 탄 버스': 'wrong bus', '가족 간 감사와 선물': 'family gratitude and gifts', 
#         '모자 구입': 'hat purchase', '현금 송금': 'cash transfer', '댄스 머신 추천': 'dance machine recommendation', '미래 직업에 대한 토론': 'discussion about future jobs', 
#         '로스앤젤레스행 비행기 시간 문의': 'inquiring about flight time to Los Angeles', '커피와 파이': 'coffee and pie', '편지 쓰기': 'writing a letter', 
#         '거주지 찾기': 'finding a residence', '슈퍼마켓 쇼핑': 'supermarket shopping', '수강 과목 선택': 'course selection', 
#         '친구 간의 전화 대화': 'phone conversation between friends', '노동자 시위': 'worker protest', '전쟁 영웅과의 대화': 'conversation with a war hero', 
#         '타투 논의': 'tattoo discussion', '무릎 부상 회복': 'knee injury recovery', '점심시간 업무 조정': 'lunchtime work coordination', 
#         '데이트 후 관계 발전': 'relationship development after a date', '객실 체크인': 'room check-in', '추수감사절 준비': 'thanksgiving preparation', 
#         '스포츠 경주 관람': 'watching a sports race', '컨퍼런스 평가': 'conference evaluation', '향수 구매': 'perfume purchase', 
#         '더위와 전기요금 문제': 'heat and electricity bill issues', '국제 관계 강연': 'lecture on international relations', '아파트 찾기': 'apartment hunting', 
#         '호텔 4층 부재': 'absence of a 4th floor in a hotel', '결혼 생활 갈등': 'marital conflict', '위치 탐색': 'location search', 
#         '크리스마스 준비': 'christmas preparation', '외국 영화 문제': 'foreign film issues', '환전': 'currency exchange', 
#         '대학 근처 아파트 임대': 'renting an apartment near campus', '식당에서의 계산': 'paying at a restaurant', '면접 절차': 'interview procedure', 
#         '아파트 평가': 'apartment evaluation', '사무실 업무': 'office work', '수영 계획': 'swimming plans', '연결 항공편 안내': 'connecting flight information', 
#         '감기 증상 및 처방': 'cold symptoms and prescription', '계좌 개설 및 안내': 'account opening and guidance', '상품 주문 및 배송 일정': 'product order and delivery schedule', 
#         '전공 변경': 'changing major', '해외에서의 돈 분실 대처': 'dealing with lost money abroad', '회의 지연으로 인한 사과': 'apology for meeting delay', 
#         '새로 생긴 가게 방문': 'visiting a new store', '진로 고민': 'career concerns', '외출 준비': 'preparing to go out', '프로젝트 논의': 'project discussion', 
#         '자동차 구매': 'car purchase', '깜짝 파티 계획': 'surprise party plan', '결혼식 준비': 'wedding preparation', '음성 메시지 설정': 'voicemail setup', 
#         '점심 회의 준비': 'preparing for a lunch meeting', '노동조합': 'labor union', '국가의 외형과 실질적 문제': "a nation's appearance vs. real issues", 
#         '사촌 Monik에 대한 이야기': 'story about cousin Monik', '옷 구매': 'buying clothes', '셀프 주유소 이용 방법': 'how to use a self-service gas station', 
#         '책 직접 홍보': 'self-promoting a book', '여름 여행 계획': 'summer travel plan', '스포츠 가방 분실': 'lost sports bag', 
#         '반려동물 동반 여행': 'traveling with a pet', '자동차 대출 상담': 'car loan consultation', '도난 신고': 'reporting a theft', 
#         '공부와 일': 'study and work', '강도 사건 목격자': 'witness to a robbery', '몰몬교 신앙': 'mormon faith', '연회 복장': 'banquet attire', 
#         '파티 춤 초대': 'party dance invitation', '세계 공통 언어': 'world common language', '아울렛 쇼핑': 'outlet shopping', 
#         '지진 뉴스와 기부': 'earthquake news and donation', '컴퓨터 구매 및 언어 지원': 'computer purchase and language support', '약속 변경': 'changing an appointment', 
#         '영업 관리자의 월급': "sales manager's salary", '가격 협상과 사무소 설치': 'price negotiation and office setup', '졸업반 반장 출마 논의': 'discussing running for senior class president', 
#         '해양 생물의 진화 강의': 'lecture on marine life evolution', '백화점 영업 시간 문의': 'department store hours inquiry', '개인실 요청': 'request for a private room', 
#         '새 집 찾기': 'finding a new house', '원예 동호회 초대': 'gardening club invitation', '만남과 인사': 'meeting and greeting', 
#         '파티에서의 인사': 'greetings at a party', '시험 준비': 'exam preparation', '생일 파티와 과제 제출': 'birthday party and assignment submission', 
#         '미국 자동차 산업': 'american auto industry', '미술 전시회 초대': 'art exhibition invitation', '헬스클럽 가입': 'joining a health club', 
#         '계절 선호': 'season preference', '음반 매장 탐색': 'exploring a record store', '콘서트 준비와 자전거 수리': 'concert prep and bike repair', 
#         '리포트 교정': 'proofreading a report', '결혼 여부': 'marital status', '사내 연애': 'office romance', '문화적 차이와 중국인 습관': 'cultural differences and chinese habits', 
#         '재킷 구매': 'buying a jacket', '어학 강좌': 'language course', '정원 가꾸기': 'gardening', '어머니날 선물 설문조사': "mother's day gift survey", 
#         '동생에게 장난': 'playing a prank on a sibling', '진실을 말하기': 'telling the truth', '개 짖음으로 인한 이웃 간의 갈등': 'neighbor conflict due to barking dog', 
#         '아이의 그림 설명': "child's drawing explanation", '식당 일시 휴업': 'restaurant temporarily closed', '독일 인사': 'german greetings', 
#         '잔돈 교환 요청': 'request to exchange change', '회사 직원 찾기': 'looking for a company employee', '캠브리지로 가는 길 안내': 'directions to Cambridge', 
#         '학위 취득 및 언어 능력': 'degree acquisition and language ability', '아마추어 산악인': 'amateur mountaineer', '사이즈 교환': 'size exchange', 
#         '의사 상담': 'doctor consultation', '식중독 의심': 'suspected food poisoning', '헬렌의 생일 파티': "helen's birthday party", 
#         '단원 마무리 시험': 'end-of-unit test', '식당에서의 대화': 'conversation at a restaurant', '영어 관용구 설명': 'explaining an english idiom', 
#         '인터뷰 후 대화': 'post-interview conversation', '학교 편입': 'school transfer', '여행 계획 취소': 'trip cancellation', 
#         '영화 관람 계획': 'movie viewing plans', '아파트 임대 상담': 'apartment rental consultation', '새로운 직장 환경': 'new work environment', 
#         '현금 출금 요청': 'cash withdrawal request', '수업 시간 조정': 'class time adjustment', '항공권 예약 확인': 'flight ticket confirmation', 
#         '평생 교육': 'lifelong education', '택시 요청': 'requesting a taxi', '농구 경기': 'basketball game', '주거비 고민': 'housing cost worries', 
#         '기온과 계절 선호': 'temperature and season preference', '부모님과의 휴가 계획': 'vacation plans with parents', '오바마 대통령 당선': 'president obama election', 
#         '버스 타다가 부딪힘': 'bumping on the bus', '호텔 예약 및 방문 계획': 'hotel reservation and visit plan', '목표 설정과 달성': 'goal setting and achievement', 
#         '여름 일자리 구직': 'summer job hunting', '베이징 오페라 관람': 'watching beijing opera', '콘서트 계획': 'concert plans', '점심 장소 추천': 'lunch spot recommendation', 
#         '새 직원 배치': 'new employee placement', '사무실 벨소리 문제': 'office ringtone problem', '뉴스 소비 방식': 'news consumption habits', 
#         '신입생 인터뷰': 'freshman interview', '남편의 가사 활동': "husband's housework", '새로운 쇼': 'new show', '공항에서의 지연': 'delay at the airport', 
#         '택시 호출': 'hailing a taxi', '채용 인터뷰': 'hiring interview', '영화 산업 변화': 'changes in the film industry', '종교적 믿음': 'religious beliefs', 
#         '농구 경기 전략': 'basketball game strategy', '이사회 회의 안건': 'board meeting agenda', '새해 결심': "new year's resolution", 
#         '끔찍한 일주일': 'terrible week', '출산 후 회복': 'postpartum recovery', '기말시험 준비': 'final exam preparation', '진로 상담': 'career counseling', 
#         '저녁 식사 예약': 'dinner reservation', '아파트 임대': 'apartment rental', '레스토랑 주문': 'restaurant order', '동물원 가는 길 안내': 'directions to the zoo', 
#         '부재자 투표': 'absentee voting', '신용카드 분실': 'lost credit card', '광고의 영향': 'influence of advertising', '신약 실험': 'new drug trial', 
#         '새 직장 시작': 'starting a new job', '계획': 'plan', '주말 활동': 'weekend activities', '건강 검진 상담': 'health checkup consultation', 
#         '휴가 경험': 'vacation experience', '고된 운동 수업': 'tough exercise class', '차량 동승': 'carpooling', '퀴즈 쇼 승리': 'winning a quiz show', 
#         '케이터링 예약': 'catering reservation', '런던 이주와 직장 생활': 'moving to london and work life', '휴가 중 건강 문제': 'health problems during vacation', 
#         '호텔 예약 및 결제': 'hotel reservation and payment', '약속 조율': 'coordinating appointments', '스페인어 공부와 간호학과 지원': 'studying spanish and applying to nursing school', 
#         '모닝콜 설정': 'setting a morning call', '작가 인터뷰': 'author interview', '송별 만찬': 'farewell dinner', '수업 결석 문제': 'class absence issue', 
#         '출장 준비': 'business trip preparation', '기차표 구매': 'train ticket purchase', '수면 시간 논쟁': 'sleep time argument', 
#         '치통 상담': 'toothache consultation', '앵무새 대화': 'parrot conversation', '첨단 기술 제품의 판매': 'sales of high-tech products', 
#         '영화 장르 선호': 'movie genre preference', '이별': 'breakup', '온라인 쇼핑': 'online shopping', '프로젝트 지원 요청': 'project support request', 
#         '화장품 구매': 'cosmetics purchase', '베이징 패키지 여행': 'beijing package tour', '이론에 대한 논쟁': 'debate over a theory', 
#         '클래식 음악 감상': 'listening to classical music', '친절한 동네 사람들': 'friendly neighbors', '에어컨 온도 조절 문제': 'air conditioner temperature issue', 
#         '일과 공부의 균형': 'work-study balance', '저녁 주문': 'dinner order', '거래 조건 협상': 'negotiating terms of a deal', '만두 만들기': 'making dumplings', 
#         '호텔 숙박 경험': 'hotel stay experience', '약속 장소 조율': 'coordinating a meeting place'
#     }
#     # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


#     df['english_topic'] = df['topic'].map(topic_map)
#     df['english_topic'].fillna('unknown', inplace=True)
#     df['topic_token'] = '<' + df['english_topic'].str.replace(' ', '_') + '>'
#     print(f"정제 및 전처리 후 데이터 크기: {len(df)}개")

#     # --- 3. 최종 컬럼 선택 및 데이터셋 분할 ---
#     final_df = df[['english_dialogue', 'english_summary', 'topic_token']]

#     print("데이터셋을 훈련용과 검증용으로 분할합니다...")
#     train_df, val_df = train_test_split(final_df, test_size=0.1, random_state=42, shuffle=True)

#     # --- 4. 최종 파일 저장 ---
#     train_output_path = os.path.join(project_root, 'data', 'processed', 'train.csv')
#     val_output_path = os.path.join(project_root, 'data', 'processed', 'val.csv')

#     train_df.to_csv(train_output_path, index=False, encoding='utf-8-sig')
#     val_df.to_csv(val_output_path, index=False, encoding='utf-8-sig')

#     print("\n✅ 훈련/검증 데이터셋 저장 완료!")
#     print(f"  - 훈련셋 경로: {train_output_path} ({len(train_df)}개)")
#     print(f"  - 검증셋 경로: {val_output_path} ({len(val_df)}개)")

# if __name__ == "__main__":
#     main()