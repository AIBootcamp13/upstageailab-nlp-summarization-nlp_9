import pytorch_lightning as pl
from transformers import (
    BartForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
import torch


class SummaryModelModule(pl.LightningModule):
    """
    PyTorch Lightning의 모델 모듈.
    모델 정의, 학습/검증 단계, 옵티마이저 설정을 담당합니다.
    """

    def __init__(self, model_cfg, tokenizer, **kwargs):
        super().__init__()
        # 설정을 저장하여 나중에 쉽게 불러올 수 있게 합니다.
        # tokenizer는 객체이므로 저장하지 않습니다.
        self.save_hyperparameters(ignore=["tokenizer"])

        # 모델 로딩 (베이스라인의 load_tokenizer_and_model_for_train 함수 역할)
        self.model = BartForConditionalGeneration.from_pretrained(
            self.hparams.model_cfg.pretrained_model_name_or_path
        )
        # DataModule에서 special token이 추가된 tokenizer의 길이에 맞게
        # 모델의 임베딩 크기를 조정합니다.
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, **inputs):
        """모델의 순전파를 정의합니다."""
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        """학습 단계를 정의합니다."""
        # 모델의 forward pass를 실행하고 loss를 얻습니다.
        # BartForConditionalGeneration은 'labels'가 제공되면 자동으로 loss를 계산합니다.
        outputs = self(**batch)
        loss = outputs.loss

        # 학습 손실을 'train_loss'라는 이름으로 로깅합니다.
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """검증 단계를 정의합니다."""
        outputs = self(**batch)
        loss = outputs.loss

        # 검증 손실을 'val_loss'라는 이름으로 로깅합니다.
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """옵티마이저와 학습률 스케줄러를 설정합니다."""
        # 옵티마이저 설정 (베이스라인의 AdamW 사용)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.model_cfg.learning_rate,
            weight_decay=self.hparams.model_cfg.weight_decay,
        )

        # 학습률 스케줄러 설정 (베이스라인의 Cosine 스케줄러 사용)
        # PyTorch Lightning 2.0 이상에서는 estimated_stepping_batches를 사용합니다.
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * self.hparams.model_cfg.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]