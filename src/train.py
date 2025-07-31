# src/train.py
import os 
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

# from src.data_module import SummaryDataModule
# from src.model_module import SummaryModelModule
from .data_module import SummaryDataModule
from .model_module import SummaryModelModule

# 하이드라가 실행 시마다 작업 디렉토리를 변경하기 때문에, 경로를 고정시켜주는 것이 안정적이다.
@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Hydra와 PyTorch Lightning을 사용하여 모델 학습을 수행하는 메인 함수.

    Args:
        cfg (DictConfig): Hydra를 통해 로드된 설정 객체.
    """
    # (성능 향상) Tensor Core를 사용하기 위한 설정
    torch.set_float32_matmul_precision('high')

    # 재현성을 위한 시드 설정
    pl.seed_everything(cfg.seed)

    # 1. 데이터 모듈 초기화
    data_module = SummaryDataModule(cfg.data, cfg.model)

    # 2. 모델 모듈 초기화 (DataModule에서 생성된 토크나이저 전달)
    model_module = SummaryModelModule(
        model_cfg=cfg.model, # 모델과 옵티마이저 설정을 함께 전달
        tokenizer=data_module.tokenizer,
    )

    # 3. 로거 및 콜백, 트레이너 초기화
    wandb_logger = hydra.utils.instantiate(cfg.wandb)

    # ModelCheckpoint 콜백의 저장 경로를 프로젝트 루트 기준의 절대 경로로 설정합니다.
    # Hydra가 실행 시마다 작업 디렉토리를 변경하기 때문에, 경로를 고정시켜주는 것이 안정적입니다.
    callbacks = []
    for callback_cfg in cfg.trainer.callbacks:
        # 설정 객체를 복사하여 원본을 유지합니다.
        _callback_cfg = callback_cfg.copy()
        if _callback_cfg["_target_"] == "pytorch_lightning.callbacks.ModelCheckpoint":
            # dirpath를 절대 경로로 변경합니다.
            original_cwd = hydra.utils.get_original_cwd() # /root/nlp
            _callback_cfg["dirpath"] = os.path.join(original_cwd, _callback_cfg["dirpath"])
        callbacks.append(hydra.utils.instantiate(_callback_cfg))

    trainer = hydra.utils.instantiate(cfg.trainer, logger=wandb_logger, callbacks=callbacks)
    
    # (모델 크기 조정) 모델의 토큰 임베딩 크기를 데이터 모듈의 토크나이저 크기에 맞춥니다.
    # 이는 토크나이저가 새로운 토큰을 추가했을 때 모델이 이를 인식할 수 있도록 합니다.
    model_module.model.resize_token_embeddings(len(data_module.tokenizer))


    # (시각화 강화) 모델의 그래디언트와 파라미터를 로깅합니다.
    wandb_logger.watch(model_module, log="all", log_freq=500)

    # 4. 학습 시작
    trainer.fit(model=model_module, datamodule=data_module)


if __name__ == "__main__":
    train()
