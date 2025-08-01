# src/train.py (Hydra 없는 최종 버전)
import os
import sys
import argparse
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl

# 상대 경로 import를 위해 프로젝트 루트를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_module import SummaryDataModule
from src.model_module import SummaryModelModule

def train(cfg):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(cfg.seed)
    data_module = SummaryDataModule(cfg.data, cfg.model)
    model_module = SummaryModelModule(cfg.model, data_module.tokenizer)

    # wandb 로거 설정
    wandb_conf = {k: v for k, v in cfg.wandb.items() if k != '_target_'}
    wandb_logger = pl.loggers.WandbLogger(**wandb_conf)

    # 콜백 설정
    callbacks = []
    for callback_conf in cfg.trainer.callbacks:
        target_class = callback_conf["_target_"]
        conf = {k:v for k,v in callback_conf.items() if k != '_target_'}
        if target_class == "pytorch_lightning.callbacks.ModelCheckpoint":
            # dirpath 포맷팅 적용
            conf["dirpath"] = conf["dirpath"].format(model_name=cfg.model.name)
            callbacks.append(pl.callbacks.ModelCheckpoint(**conf))
        elif target_class == "pytorch_lightning.callbacks.EarlyStopping":
            callbacks.append(pl.callbacks.EarlyStopping(**conf))

    # 트레이너 설정
    trainer_args = {k:v for k,v in cfg.trainer.items() if k != '_target_' and k != 'callbacks'}
    trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, **trainer_args)

    model_module.model.resize_token_embeddings(len(data_module.tokenizer))
    wandb_logger.watch(model_module, log="all", log_freq=500)

    print("🚀 [Trainer] 훈련을 시작합니다...")
    trainer.fit(model=model_module, datamodule=data_module)
    print("✅ [Trainer] 훈련이 완료되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="e.g., flan-t5")
    args = parser.parse_args()

    # 설정 파일 수동 로드 및 병합
    cfg = OmegaConf.load('configs/config.yaml')
    model_cfg = OmegaConf.load(f'configs/model/{args.model_name}.yaml')
    trainer_cfg = OmegaConf.load(f"configs/trainer/default.yaml")
    data_cfg = OmegaConf.load(f"configs/data/default.yaml")

    # model, trainer, data 설정을 기본 cfg에 병합
    cfg.merge_with({'model': model_cfg, 'trainer': trainer_cfg, 'data': data_cfg})

    train(cfg)