# src/inference.py (Hydra 없는 최종 버전)

import os
import sys
import re
import glob
import argparse
# from datetime import datetime
from datetime import datetime, timezone, timedelta # <-- 이 부분을 수정/추가
import pandas as pd
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl

# 상대 경로 import를 위해 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_module import SummaryDataModule
from src.model_module import SummaryModelModule

def inference(cfg):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(cfg.seed)
    
    # DataModule은 predict stage를 위해 인스턴스화
    data_module = SummaryDataModule(cfg.data, cfg.model)
    data_module.setup(stage='predict')

    # --- 체크포인트 탐색 로직 ---
    search_dir = os.path.join(os.getcwd(), "all_checkpoints", cfg.model.name)
    ckpt_files = glob.glob(os.path.join(search_dir, "*.ckpt"))
    
    if not ckpt_files:
        raise FileNotFoundError(f"'{cfg.model.name}' 모델의 체크포인트 파일을 찾을 수 없습니다.")

    # def get_score_from_path(p):
    #     match = re.search(r"rougeL=([\d.]+)", p)
    #     return float(match.group(1)) if match else 0
    # src/inference.py 의 get_score_from_path 함수를 교체

    def get_score_from_path(p):
        match = re.search(r"rougeL=([\d.]+)", p)
        if match:
            # 끝에 있을지 모르는 점(.)을 제거하여 "0.0000." 같은 경우를 처리
            score_str = match.group(1).rstrip('.')
            return float(score_str)
        return 0

    best_ckpt_path = max(ckpt_files, key=get_score_from_path)
    print(f"'{cfg.model.name}' 모델의 가장 좋은 체크포인트를 로드합니다: {best_ckpt_path}")

    # 체크포인트로부터 모델 모듈 로드
    model_module = SummaryModelModule.load_from_checkpoint(
        best_ckpt_path,
        model_cfg=cfg.model,
        tokenizer=data_module.tokenizer
    )

    model_module.eval()
    model_module.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # 커맨드 라인 generation 설정을 모델에 주입
    if 'generation' in cfg:
        model_module.hparams.generation = cfg.generation
        print(f"추론 옵션을 적용합니다: {OmegaConf.to_container(cfg.generation)}")
    
    trainer = pl.Trainer(accelerator="auto", devices=1)
    
    print("🚀 추론을 시작합니다...")
    predictions = trainer.predict(model=model_module, dataloaders=data_module.predict_dataloader())
    print("✅ 추론이 완료되었습니다.")

    # 결과 취합 및 제출 파일 생성
    all_summaries = [summary for batch in predictions for summary in batch]
    test_df_path = os.path.join(os.getcwd(), 'data', 'raw', 'test.csv')
    test_df = pd.read_csv(test_df_path)
    submission = pd.DataFrame({'fname': test_df['fname'], 'summary': all_summaries})
    
    submissions_dir = "submissions"
    os.makedirs(submissions_dir, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ▼▼▼▼▼ 타임스탬프 생성 부분을 아래 코드로 교체 ▼▼▼▼▼
    # KST 시간대 (UTC+9) 정의
    kst = timezone(timedelta(hours=9))
    # 현재 시간을 KST 기준으로 생성
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        
    submission_path = os.path.join(submissions_dir, f'submission_{timestamp}.csv')
    
    # 대회 규정에 맞게 index=False로 저장
    submission.to_csv(submission_path, index=True, encoding='utf-8-sig')
    print(f"제출 파일이 '{submission_path}'에 성공적으로 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="e.g., flan-t5-large")
    # `num_beams`와 같은 추론 옵션을 받기 위한 설정
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--length_penalty", type=float, default=None)
    args = parser.parse_args()

    # 설정 파일 수동 로드 및 병합
    cfg = OmegaConf.load('configs/config.yaml')
    model_cfg = OmegaConf.load(f'configs/model/{args.model_name}.yaml')
    trainer_cfg = OmegaConf.load(f"configs/trainer/default.yaml")
    data_cfg = OmegaConf.load(f"configs/data/default.yaml")
    cfg.merge_with({'model': model_cfg, 'trainer': trainer_cfg, 'data': data_cfg})
    
    # 커맨드 라인 인자를 generation 설정으로 추가
    generation_cfg = {
        'num_beams': args.num_beams,
        'length_penalty': args.length_penalty
    }
    cfg.merge_with({'generation': generation_cfg})
    
    inference(cfg)