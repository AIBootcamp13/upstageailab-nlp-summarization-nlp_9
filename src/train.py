import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.data_module import SummaryDataModule
from src.model_module import SummaryModelModule


@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    Hydra와 PyTorch Lightning을 사용하여 모델 학습을 수행하는 메인 함수.

    Args:
        cfg (DictConfig): Hydra를 통해 로드된 설정 객체.
    """
    # 재현성을 위한 시드 설정
    pl.seed_everything(cfg.seed)

    # 1. 데이터 모듈 초기화
    data_module = SummaryDataModule(cfg.data, cfg.model)

    # 2. 모델 모듈 초기화 (DataModule에서 생성된 토크나이저 전달)
    model_module = SummaryModelModule(
        model_cfg=cfg.model, # 모델과 옵티마이저 설정을 함께 전달
        tokenizer=data_module.tokenizer,
    )

    # 3. 로거 및 트레이너 초기화 (Hydra가 자동으로 설정 객체를 클래스 인스턴스로 변환)
    wandb_logger = hydra.utils.instantiate(cfg.wandb)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=wandb_logger)

    # 4. 학습 시작
    trainer.fit(model=model_module, datamodule=data_module)


if __name__ == "__main__":
    train()
