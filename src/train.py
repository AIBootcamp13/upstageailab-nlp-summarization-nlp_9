# src/train.py (최종 훈련용 깨끗한 버전)
import os
import sys
import argparse
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl

# 경로와 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_module import SummaryDataModule
from src.model_module import SummaryModelModule

def train(cfg):
    # 기본 설정
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(cfg.seed)
    
    # 모듈 및 로거 초기화
    wandb_logger = pl.loggers.WandbLogger(project=cfg.wandb.project)
    data_module = SummaryDataModule(cfg.data, cfg.model)
    model_module = SummaryModelModule(cfg.model, data_module.tokenizer)

    # 콜백 설정
    callbacks = []
    for callback_conf in cfg.trainer.callbacks:
        if callback_conf["_target_"] == "pytorch_lightning.callbacks.ModelCheckpoint":
            conf = {k:v for k,v in callback_conf.items() if k != '_target_'}
            callbacks.append(pl.callbacks.ModelCheckpoint(**conf))
        elif callback_conf["_target_"] == "pytorch_lightning.callbacks.EarlyStopping":
            conf = {k:v for k,v in callback_conf.items() if k != '_target_'}
            callbacks.append(pl.callbacks.EarlyStopping(**conf))

    # 트레이너 설정
    trainer_args = {k:v for k,v in cfg.trainer.items() if k != '_target_' and k != 'callbacks'}
    trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, **trainer_args)
    
    # special_tokens 설정이 있을 때만 resize 실행 (현재는 null이라 실행 안 됨)
    if cfg.model.get("special_tokens"):
        model_module.model.resize_token_embeddings(len(data_module.tokenizer))
    
    wandb_logger.watch(model_module, log="all", log_freq=500)
    
    # 훈련 시작
    print(f"🚀 [Trainer] 최종 훈련을 시작합니다... 모델: {cfg.model.name}")
    trainer.fit(model=model_module, datamodule=data_module)
    print("✅ [Trainer] 최종 훈련이 완료되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="e.g., flan-t5-large-final")
    args, unknown_args = parser.parse_known_args()

    # 설정 파일 로드 및 병합
    cfg = OmegaConf.load('configs/config.yaml')
    model_cfg = OmegaConf.load(f'configs/model/{args.model_name}.yaml')
    trainer_cfg = OmegaConf.load(f"configs/trainer/default.yaml")
    data_cfg = OmegaConf.load(f"configs/data/default.yaml")
    cfg.merge_with({'model': model_cfg, 'trainer': trainer_cfg, 'data': data_cfg})
    
    # 커맨드 라인 오버라이드 적용
    if unknown_args:
        cli_cfg = OmegaConf.from_dotlist(unknown_args)
        cfg.merge_with(cli_cfg)
        
    train(cfg)
    
# # src/train.py (Hydra 없는 최종 버전)
# import os
# import sys
# import argparse
# from omegaconf import OmegaConf
# import torch
# import pytorch_lightning as pl

# # 상대 경로 import를 위해 프로젝트 루트를 sys.path에 추가
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.data_module import SummaryDataModule
# from src.model_module import SummaryModelModule

# def train(cfg):
#     torch.set_float32_matmul_precision('high')
#     pl.seed_everything(cfg.seed)
#     data_module = SummaryDataModule(cfg.data, cfg.model)
#     model_module = SummaryModelModule(cfg.model, data_module.tokenizer)

#     # wandb 로거 설정
#     wandb_conf = {k: v for k, v in cfg.wandb.items() if k != '_target_'}
#     wandb_logger = pl.loggers.WandbLogger(**wandb_conf)

#     # 콜백 설정
#     callbacks = []
#     for callback_conf in cfg.trainer.callbacks:
#         target_class = callback_conf["_target_"]
#         conf = {k:v for k,v in callback_conf.items() if k != '_target_'}
#         if target_class == "pytorch_lightning.callbacks.ModelCheckpoint":
#             # dirpath 포맷팅 적용
#             conf["dirpath"] = conf["dirpath"].format(model_name=cfg.model.name)
#             callbacks.append(pl.callbacks.ModelCheckpoint(**conf))
#         elif target_class == "pytorch_lightning.callbacks.EarlyStopping":
#             callbacks.append(pl.callbacks.EarlyStopping(**conf))

#     # 트레이너 설정
#     trainer_args = {k:v for k,v in cfg.trainer.items() if k != '_target_' and k != 'callbacks'}
#     trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, **trainer_args)

#     model_module.model.resize_token_embeddings(len(data_module.tokenizer))
#     wandb_logger.watch(model_module, log="all", log_freq=500)

#     print("🚀 [Trainer] 훈련을 시작합니다...")
#     trainer.fit(model=model_module, datamodule=data_module)
#     print("✅ [Trainer] 훈련이 완료되었습니다.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", type=str, required=True, help="e.g., flan-t5")
#     args = parser.parse_args()

#     # 설정 파일 수동 로드 및 병합
#     cfg = OmegaConf.load('configs/config.yaml')
#     model_cfg = OmegaConf.load(f'configs/model/{args.model_name}.yaml')
#     trainer_cfg = OmegaConf.load(f"configs/trainer/default.yaml")
#     data_cfg = OmegaConf.load(f"configs/data/default.yaml")

#     # model, trainer, data 설정을 기본 cfg에 병합
#     cfg.merge_with({'model': model_cfg, 'trainer': trainer_cfg, 'data': data_cfg})

#     train(cfg)
# src/train.py (Sweep 전용 최종 버전)
# import os
# import sys
# from omegaconf import OmegaConf
# import torch
# import pytorch_lightning as pl
# import wandb

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.data_module import SummaryDataModule
# from src.model_module import SummaryModelModule

# def train(cfg):
#     wandb_logger = pl.loggers.WandbLogger(project=cfg.wandb.project)
#     torch.set_float32_matmul_precision('high')
#     pl.seed_everything(cfg.seed)
#     data_module = SummaryDataModule(cfg.data, cfg.model)
#     model_module = SummaryModelModule(cfg.model, data_module.tokenizer)
#     callbacks = []
#     for callback_conf in cfg.trainer.callbacks:
#         if callback_conf["_target_"] == "pytorch_lightning.callbacks.ModelCheckpoint":
#             conf = {k:v for k,v in callback_conf.items() if k != '_target_'}
#             callbacks.append(pl.callbacks.ModelCheckpoint(**conf))
#         elif callback_conf["_target_"] == "pytorch_lightning.callbacks.EarlyStopping":
#             conf = {k:v for k,v in callback_conf.items() if k != '_target_'}
#             callbacks.append(pl.callbacks.EarlyStopping(**conf))
#     trainer_args = {k:v for k,v in cfg.trainer.items() if k != '_target_' and k != 'callbacks'}
#     trainer = pl.Trainer(logger=wandb_logger, callbacks=callbacks, **trainer_args)
#     if cfg.model.get("special_tokens"): # special_tokens 설정이 있을 때만 실행
#         model_module.model.resize_token_embeddings(len(data_module.tokenizer))
#     wandb_logger.watch(model_module, log="all", log_freq=500)
#     print("🚀 [Trainer] 훈련을 시작합니다...")
#     trainer.fit(model=model_module, datamodule=data_module)
#     print("✅ [Trainer] 훈련이 완료되었습니다.")

# if __name__ == "__main__":
#     wandb.init()
#     sweep_config = wandb.config
#     cfg = OmegaConf.load('configs/config.yaml')
#     model_cfg = OmegaConf.load(f'configs/model/{sweep_config.model_name}.yaml')
#     trainer_cfg = OmegaConf.load(f"configs/trainer/default.yaml")
#     data_cfg = OmegaConf.load(sweep_config.data_config_path)
#     model_cfg.learning_rate = sweep_config.learning_rate
#     if hasattr(sweep_config, 'weight_decay'):
#         model_cfg.weight_decay = sweep_config.weight_decay
#     if hasattr(sweep_config, 'label_smoothing'):
#         model_cfg.label_smoothing = sweep_config.label_smoothing
#     trainer_cfg.max_epochs = sweep_config.max_epochs
#     cfg.merge_with({'model': model_cfg, 'trainer': trainer_cfg, 'data': data_cfg})
#     train(cfg)