# src/validate.py
# 최고의 num_beams 값 찾기
import os
import sys
import re
import glob
import argparse
import pandas as pd
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
import evaluate  # evaluate 라이브러리 추가

# 상대 경로 import를 위해 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_module import SummaryDataModule
from src.model_module import SummaryModelModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def validate(cfg):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(cfg.seed)
    
    # DataModule을 'fit' 스테이지로 설정하여 val_dataloader를 준비
    data_module = SummaryDataModule(cfg.data, cfg.model)
    data_module.setup(stage='fit')

    # --- 체크포인트 탐색 로직 ---
    search_dir = os.path.join(os.getcwd(), "all_checkpoints", cfg.model.name)
    ckpt_files = glob.glob(os.path.join(search_dir, "*.ckpt"))
    
    if not ckpt_files:
        raise FileNotFoundError(f"'{cfg.model.name}' 모델의 체크포인트 파일을 찾을 수 없습니다.")

    def get_score_from_path(p):
        # ROUGE-L 점수를 찾도록 정규표현식 수정
        match = re.search(r"rougeL=([\d.]+)", p)
        if match:
            # 끝에 있을지 모르는 점(.)을 제거하여 "0.0000." 같은 경우를 처리
            score_str = match.group(1).rstrip('.')
            return float(score_str)
        return 0

    best_ckpt_path = max(ckpt_files, key=get_score_from_path)
    print(f"'{cfg.model.name}' 모델의 가장 좋은 체크포인트를 로드합니다: {best_ckpt_path}")

    model_module = SummaryModelModule.load_from_checkpoint(
        best_ckpt_path,
        model_cfg=cfg.model,
        tokenizer=data_module.tokenizer
    )

    model_module.eval()
    model_module.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if 'generation' in cfg:
        model_module.hparams.generation = cfg.generation
        print(f"추론 옵션을 적용합니다: {OmegaConf.to_container(cfg.generation)}")
    
    trainer = pl.Trainer(accelerator="auto", devices=1)
    
    print("🚀 검증 데이터로 추론을 시작합니다...")
    # datamodule 대신 val_dataloader를 직접 전달
    predictions = trainer.predict(model=model_module, dataloaders=data_module.val_dataloader())
    print("✅ 추론이 완료되었습니다.")

    # --- ROUGE 점수 계산 로직 추가 ---
    print("💯 ROUGE 점수를 계산합니다...")
    all_summaries = [summary for batch in predictions for summary in batch]
    
    # val.csv 파일에서 정답 요약문(references) 불러오기
    val_df = pd.read_csv(cfg.data.val_path)
    references = val_df['english_summary'].tolist()

    # ROUGE 계산기 로드 및 점수 계산
    rouge_metric = evaluate.load("rouge")
    rouge_metric.add_batch(predictions=all_summaries, references=references)
    results = rouge_metric.compute()

    print("\n--- 검증 결과 (ROUGE Scores) ---")
    print(results)
    print("---------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="e.g., flan-t5-large")
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--length_penalty", type=float, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load('configs/config.yaml')
    model_cfg = OmegaConf.load(f'configs/model/{args.model_name}.yaml')
    trainer_cfg = OmegaConf.load(f"configs/trainer/default.yaml")
    data_cfg = OmegaConf.load(f"configs/data/default.yaml")
    cfg.merge_with({'model': model_cfg, 'trainer': trainer_cfg, 'data': data_cfg})
    
    generation_cfg = {
        'num_beams': args.num_beams,
        'length_penalty': args.length_penalty
    }
    cfg.merge_with({'generation': generation_cfg})
    
    validate(cfg)