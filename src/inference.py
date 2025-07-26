# src/inference.py

import os
import hydra
import pandas as pd
import glob
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.base import ContainerMetadata
import pytorch_lightning as pl

# PyTorch 버전 호환성을 위한 설정
# PyTorch 2.6+에서는 add_safe_globals, 이전 버전에서는 package_importer를 사용
try:
    # PyTorch 2.6+
    from torch.serialization import add_safe_globals
    add_safe_globals([
        OmegaConf.get_type(DictConfig),
        ContainerMetadata,
    ])
except ImportError:
    # PyTorch 2.6 미만
    from torch.package import package_importer
    @package_importer.importer
    def import_omegaconf(name):
        if name == 'omegaconf.dictconfig':
            return OmegaConf.get_type(DictConfig)
        if name == 'omegaconf.container':
            return ContainerMetadata

from src.data_module import SummaryDataModule
from src.model_module import SummaryModelModule


@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def inference(cfg: DictConfig) -> None:
    """
    학습된 모델로 추론을 수행하고 submission.csv 파일을 생성합니다.

    Args:
        cfg (DictConfig): Hydra를 통해 로드된 설정 객체.
    """
    # GPU/TPU 등 장치 설정
    torch.set_float32_matmul_precision('high')

    # 1. 데이터 모듈 초기화
    data_module = SummaryDataModule(cfg.data, cfg.model)

    # 2. 가장 최근에 저장된 체크포인트 경로 탐색
    # 모든 체크포인트가 저장되는 'all_checkpoints' 폴더에서 탐색합니다.
    # 이렇게 하면 Hydra의 실행별 출력 폴더 구조에 영향을 받지 않아 안정적입니다.
    search_path = os.path.join(hydra.utils.get_original_cwd(), "all_checkpoints/*.ckpt")
    ckpt_files = glob.glob(search_path)
    if not ckpt_files:
        raise FileNotFoundError("체크포인트 파일을 'all_checkpoints' 폴더에서 찾을 수 없습니다. 먼저 학습을 실행하여 체크포인트를 생성하세요.")
    
    latest_ckpt_path = max(ckpt_files, key=os.path.getmtime)
    print(f"가장 최근 체크포인트를 로드합니다: {latest_ckpt_path}")

    # 3. 체크포인트로부터 모델 모듈 로드
    # PyTorch 2.6 이상 버전의 weights_only=True 기본값 정책으로 인해
    # lightning의 load_from_checkpoint 대신 torch.load를 사용하여 직접 로드합니다.
    model_module = SummaryModelModule(cfg.model, data_module.tokenizer)
    checkpoint = torch.load(latest_ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    model_module.load_state_dict(checkpoint['state_dict'])

    # 모델의 토큰 임베딩 크기를 데이터 모듈의 토크나이저 크기에 맞춥니다.
    model_module.model.resize_token_embeddings(len(data_module.tokenizer))

    # 4. 트레이너 초기화
    trainer = hydra.utils.instantiate(cfg.trainer)

    # 5. 추론 실행 (모델은 이미 로드되었으므로 ckpt_path 인자는 필요 없습니다)
    print("Starting inference...")
    predictions = trainer.predict(model=model_module, datamodule=data_module)
    print("Inference finished.")

    # 6. 결과 취합 및 제출 파일 생성
    all_summaries = [summary for batch in predictions for summary in batch]
    test_df = pd.read_csv(os.path.join(cfg.data.path, 'test.csv'))
    submission = pd.DataFrame({'fname': test_df['fname'], 'summary': all_summaries})
    
    # submissions 폴더 생성
    os.makedirs("submissions", exist_ok=True)
    
    # 파일명에 타임스탬프 추가
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = f'submissions/submission_{timestamp}.csv'
    
    submission.to_csv(submission_path, index=True)
    print(f"Submission file '{submission_path}' created successfully!")
    print("Submission file 'submission.csv' created successfully!")

if __name__ == "__main__":
    inference()
