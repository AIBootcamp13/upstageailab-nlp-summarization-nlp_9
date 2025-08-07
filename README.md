# Dialogue Summarization | 일상 대화 요약 경진대회

## Team : CV-Team9

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | 
|            [홍정민](https://github.com/UpstageAILab)             |            [최지희](https://github.com/UpstageAILab)             |            [이재용](https://github.com/UpstageAILab)             |            [김효석](https://github.com/UpstageAILab)             |
|                            팀장, 데이터 전처리/모델학습                             |                            데이터 전처리/모델학습                             |                            데이터 전처리/모델학습                             |                            데이터 전처리/모델학습                             |

## 0. Overview

### Environment
- OS: Ubuntu / CUDA
- Python 3.10
- GPU: Tesla T4 / A100

### Requirements
- transformers
- datasets
- peft
- bitsandbytes
- accelerate
- deep-translator

---

## 1. Competition Info

### Overview
- 실제 일상 시나리오 기반 대화를 한 문장으로 요약하는 모델 개발 대회
- 다양한 언어모델을 적용하여 성능 향상 전략을 실험함

### Timeline
- 2025.07.25~ 2025.08.06

---

## 2. Components

### Directory

```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

---

## 3. Data Description

### Dataset overview
- 총 13,455개의 한국어 대화 데이터
- 구성: `dialogue`, `summary`, `topic`
- train: 12,457 / validation: 499 / test: 499

📁 train.csv / dev.csv
| Column     | 설명                       |
| ---------- | ------------------------ |
| `fname`    | 샘플 ID                    |
| `dialogue` | 다자간 일상 대화 텍스트            |
| `summary`  | 대화 내용 요약문 (정답)           |
| `topic`    | 대화 주제 (주제별 성능 분석에 사용 가능) |


📁 test.csv
| Column                    | 설명            |
| ------------------------- | ------------- |
| `fname`                   | 샘플 ID         |
| `dialogue`                | 다자간 일상 대화 텍스트 |
| `summary` 없음 → 모델이 생성해야 함 |               |

### EDA
- 발화 길이 평균: 약 550자
- summary는 대부분 1문장
- topic 다양성 존재 (건강검진, 백신 접종, 잃어버린 물건 등)

### Data Processing
- Text Cleansing : 의미 없는 특수 문자, 공백, 이모티콘 제거
- Back translation 기반 증강 추가
- Few-shot prompting 형식으로 변환

---

## 4. Modeling

### Model Description
<img width="521" height="314" alt="스크린샷 2025-08-07 오후 12 18 53" src="https://github.com/user-attachments/assets/0521b878-dbaa-42fa-a8c4-c716b3f8e325" />

| Model | 설명 |
|-------|------|
| KoBART | baseline 모델, 빠르고 안정적 |
| T5-base | 다양한 요약 스타일 대응 가능 |
| Qwen3-1.7B | instruction tuning + few-shot 대응력 강함 |
| SOLAR-10.7B-Instruct | 한국어 instruction LLM 중 최강 성능 |

### Modeling Process
- Hugging Face Transformers 기반 fine-tuning
- LoRA / QLoRA 사용으로 경량 학습 구현
- SOLAR 모델: few-shot prompting + QLoRA 학습 구조
<img width="635" height="308" alt="스크린샷 2025-08-07 오후 12 19 27" src="https://github.com/user-attachments/assets/07eb0119-4118-4a38-ba8b-d616b612d2eb" />

---

## 5. Result

### Leader Board
<img width="864" height="610" alt="image" src="https://github.com/user-attachments/assets/fe910fe5-b44d-44ca-89b5-db20f5e1dae9" />


### Presentation
- [📄 발표 자료](https://docs.google.com/presentation/d/1FBIfIUDDA-Iw6YShXsmCJV1QxOjLCJNL/edit?slide=id.p7#slide=id.p1)

### Meeting Log
- [📝 이슈 관리)](https://trello.com/b/aaaTrVD5/9%EC%A1%B0)

