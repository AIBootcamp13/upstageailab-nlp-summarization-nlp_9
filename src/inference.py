# src/inference.py (주석 제거 및 모든 기능 통합 최종 버전)
import os
import sys
import time
import glob
import re
import argparse
import pandas as pd
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
import evaluate
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# --- 1. 경로 설정 및 API 클라이언트 초기화 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_module import SummaryDataModule
from src.model_module import SummaryModelModule

# .env 파일에서 API 키 로드
load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# Solar API 클라이언트 초기화
client = OpenAI(api_key=UPSTAGE_API_KEY, base_url="https://api.upstage.ai/v1/solar")
print("✅ Solar API 클라이언트가 초기화되었습니다.")


# --- 2. API 호출 헬퍼 함수 ---
def api_request_with_rate_limiting(messages):
    try:
        response = client.chat.completions.create(
            model="solar-1-mini-chat", messages=messages, temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"🔥🔥🔥 API 호출 중 에러 발생: {e}")
        return ""
    
def translate_texts(texts, prompt_template):
    # 'ko' -> 'en' 인지, 'en' -> 'ko' 인지 자동 감지
    direction = "ko -> en" if "Korean" not in prompt_template else "en -> ko"
    print(f"🌐 [API Call] {len(texts)}개의 텍스트를 {direction} 방향으로 번역합니다...")
    
    translated = []
    start_time = time.time()
    
    for i, text in enumerate(tqdm(texts, desc=f"Translating ({direction})")):
        # 사용자가 제공한 프롬프트 템플릿에 번역할 텍스트를 삽입
        prompt = prompt_template.format(text_to_translate=text)
        messages = [{"role": "user", "content": prompt}]
        translated_text = api_request_with_rate_limiting(messages)
        translated.append(translated_text)
        
        # Rate Limiting 로직 (이전과 동일)
        if (i + 1) % 100 == 0 and len(texts) > 100:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time + 1
                print(f"  - Rate limit: {wait_time:.1f}초 대기...")
                time.sleep(wait_time)
            start_time = time.time()
            
    return translated

# --- 3. 메인 추론 함수 (번역 프롬프트 정의 및 적용) ---
def inference(cfg):
    pl.seed_everything(cfg.seed)

    data_module = SummaryDataModule(cfg.data, cfg.model)
    best_ckpt_path = find_best_checkpoint(cfg.model.name)
    print(f"✅ [모델] 가장 좋은 체크포인트를 로드합니다: {os.path.basename(best_ckpt_path)}")
    model_module = SummaryModelModule.load_from_checkpoint(
        best_ckpt_path, tokenizer=data_module.tokenizer
    )
    data_module.tokenizer = model_module.tokenizer

    print(f"🚀 [추론] '{cfg.input_path}' 파일에 대한 추론을 시작합니다...")
    input_df = pd.read_csv(cfg.input_path)
    
    is_test_run = ('test.csv' in cfg.input_path)
    
    if is_test_run:
        # --- 1차 번역 (ko -> en)을 위한 프롬프트 ---
        KO_TO_EN_PROMPT = """
        Translate the following Korean dialogue to natural, fluent English.
        Preserve the speaker tags like #Person1# exactly.

        Korean Dialogue:
        {text_to_translate}

        English Dialogue:
        """
        dialogues_to_process = translate_texts(input_df['dialogue'].tolist(), KO_TO_EN_PROMPT)
    else:
        dialogues_to_process = input_df['input_text'].tolist()

    data_module.predict_df = pd.DataFrame({"input_text": dialogues_to_process})
    data_module.setup('predict')
    
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    model_module.hparams.generation = cfg.generation
    
    predictions = trainer.predict(model=model_module, dataloaders=data_module.predict_dataloader())
    # english_summaries = [summary for batch in predictions for summary in batch]

    # if is_test_run:
    #     # --- 2차 번역 (en -> ko)을 위한 챔피언 프롬프트 (Prompt 3) ---
    #     EN_TO_KO_PROMPT = """
    #     You are a professional translator working on a Korean language dataset for AI training.
    #     Translate the following English summary into **natural, fluent, and detailed Korean**.
    #     **Instructions:**
    #     1. Keep speaker tags such as `#Person1#`, `#person2#` **exactly as they are**.
    #     2. Personal names must be translated **phonetically** (e.g., "Francis" → "프랜시스").
    #     3. Use a **formal**, full-sentence tone.
    #     4. DO NOT summarize or skip information. Be as detailed as the source.
    #     ---
    #     Please translate the following English summary:
    #     {text_to_translate}
    #     """
    #     final_summaries = translate_texts(english_summaries, EN_TO_KO_PROMPT)
    # else:
    #     final_summaries = english_summaries


    # submission = pd.DataFrame({'fname': input_df['fname'], 'summary': final_summaries})
    # os.makedirs("submissions", exist_ok=True)
    # submission.to_csv(cfg.output_path, index=False, encoding='utf-8-sig')
    # print(f"✅ [성공] 추론 결과가 '{cfg.output_path}'에 저장되었습니다.")
    # src/inference.py 의 inference 함수 내부


    # ... (predictions = trainer.predict(...) 코드 바로 다음부터)

    all_summaries = [summary for batch in predictions for summary in batch]
    print("✅ [추론] 추론이 완료되었습니다.")

    # --- 결과 저장 및 평가 (변수 이름을 all_summaries로 수정) ---
    input_df = pd.read_csv(cfg.input_path)
    
    # ▼▼▼▼▼ final_summaries -> all_summaries 로 수정 ▼▼▼▼▼
    if 'fname' in input_df.columns:
        # 최종 제출용 test.csv처럼 fname이 있는 경우
        submission = pd.DataFrame({'fname': input_df['fname'], 'summary': all_summaries})
    else:
        # 실험용 val.csv처럼 fname이 없는 경우, 임시로 인덱스를 사용
        submission = pd.DataFrame({'id': input_df.index, 'summary': all_summaries})
        print("⚠️ [정보] 입력 파일에 'fname' 컬럼이 없어 임시 id를 사용합니다.")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    os.makedirs("submissions", exist_ok=True)
    submission.to_csv(cfg.output_path, index=False, encoding='utf-8-sig')
    print(f"✅ [성공] 추론 결과가 '{cfg.output_path}'에 저장되었습니다.")

    # ▼▼▼▼▼ ROUGE 계산 부분도 english_summaries -> all_summaries 로 수정 ▼▼▼▼▼
    if cfg.calculate_rouge:
        if 'english_summary' not in input_df.columns:
            print("⚠️ 'english_summary' 컬럼이 없어 ROUGE 점수를 계산할 수 없습니다.")
            return
        print("📊 [평가] ROUGE 점수를 계산합니다...")
        rouge_metric = evaluate.load("rouge")
        references = input_df['english_summary'].tolist()
        results = rouge_metric.compute(predictions=all_summaries, references=references) # <-- 여기도 수정!
        
        print("\n--- 📝 최종 ROUGE 점수 📝 ---")
        for key, value in results.items():
            print(f"- {key}: {value:.4f}")
        print("------------------------------")

# --- 4. 헬퍼 함수들 ---
def find_best_checkpoint(model_name):
    search_dirs = [
        os.path.join(os.getcwd(), "all_checkpoints", model_name),
        os.path.join(os.getcwd(), "all_checkpoints_backup", model_name)
    ]
    ckpt_files = []
    for s_dir in search_dirs: ckpt_files.extend(glob.glob(os.path.join(s_dir, "model-*.ckpt")))
    if not ckpt_files: raise FileNotFoundError(f"'{model_name}' 모델의 체크포인트 파일을 찾을 수 없습니다.")
    def get_score(p):
        match = re.search(r"rougeL=([\d.]+)", p)
        return float(match.group(1).rstrip('.')) if match else 0
    return max(ckpt_files, key=get_score)

# --- 5. 메인 실행 블록 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_path", type=str, default="data/processed/val.csv")
    parser.add_argument("--output_path", type=str, default=f"submissions/temp_submission_{int(time.time())}.csv")
    parser.add_argument("--calculate_rouge", action='store_true')
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--length_penalty", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    args, unknown_args = parser.parse_known_args()
    cfg = OmegaConf.load('configs/config.yaml')
    model_cfg = OmegaConf.load(f'configs/model/{args.model_name}.yaml')
    data_cfg = OmegaConf.load(f"configs/data/default.yaml")
    generation_cfg = OmegaConf.from_dotlist(unknown_args)
    arg_gen_cfg = {k: v for k, v in vars(args).items() if k not in ['model_name', 'input_path', 'output_path', 'calculate_rouge']}
    for k, v in arg_gen_cfg.items():
        if v is not None: OmegaConf.update(generation_cfg, k, v)
    cfg.merge_with({'model': model_cfg, 'data': data_cfg, 'generation': generation_cfg})
    cfg.input_path = args.input_path
    cfg.output_path = args.output_path
    cfg.calculate_rouge = args.calculate_rouge
    inference(cfg)