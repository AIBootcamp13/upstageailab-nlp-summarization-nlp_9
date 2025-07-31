# src/model_module.py

import pytorch_lightning as pl
from transformers import (
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup,
)
import torch
import evaluate
import wandb

class SummaryModelModule(pl.LightningModule):
    """
    PyTorch Lightning의 모델 모듈.
    모델 정의, 학습/검증 단계, 옵티마이저 설정을 담당합니다.
    """

    def __init__(self, model_cfg, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.rouge_metric = evaluate.load("rouge")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_cfg.pretrained_model_name_or_path
        )
        
        self.model.resize_token_embeddings(len(tokenizer))
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.hparams.model_cfg.generate_max_length,
            num_beams=self.hparams.model_cfg.num_beams,
            decoder_start_token_id=self.model.config.pad_token_id
        )

        labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        generated_summaries = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        self.rouge_metric.add_batch(predictions=generated_summaries, references=labels)

        if batch_idx == 0:
            self.validation_step_outputs.append({
                "dialogue_ids": batch["input_ids"],
                "label_ids": batch["labels"],
                "generated_ids": generated_ids,
            })

    def on_validation_epoch_end(self):
        results = self.rouge_metric.compute()
        self.log_dict(results, on_epoch=True, prog_bar=True)

        if self.validation_step_outputs:
            first_batch_outputs = self.validation_step_outputs[0]
            dialogues = self.tokenizer.batch_decode(first_batch_outputs["dialogue_ids"], skip_special_tokens=True)
            ground_truths = self.tokenizer.batch_decode(first_batch_outputs["label_ids"], skip_special_tokens=True)
            generateds = self.tokenizer.batch_decode(first_batch_outputs["generated_ids"], skip_special_tokens=True)

            columns = ["Dialogue", "Ground Truth", "Generated"]
            data = [[dialogue, truth, gen] for dialogue, truth, gen in zip(dialogues, ground_truths, generateds)]
            self.logger.log_table(key="Validation Samples", columns=columns, data=data)

        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.hparams.model_cfg.generate_max_length,
            num_beams=self.hparams.model_cfg.num_beams,
            decoder_start_token_id=self.model.config.pad_token_id
        )
        decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded_preds

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(
    #         self.model.parameters(),
    #         lr=self.hparams.model_cfg.learning_rate,
    #         weight_decay=self.hparams.model_cfg.weight_decay,
    #     )
    #     return optimizer
    # src/model_module.py 의 SummaryModelModule 클래스 안에 있는

    # 웜업 스케줄러를 포함한 옵티마이저 
    def configure_optimizers(self):
        # 1. 옵티마이저 정의
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.model_cfg.learning_rate,
            weight_decay=self.hparams.model_cfg.weight_decay,
        )

        # 2. 총 학습 스텝 수 계산
        # PyTorch Lightning 2.0 이상에서는 이 방식이 가장 안정적이야.
        num_training_steps = self.trainer.estimated_stepping_batches

        # 3. 웜업 스텝 수 계산 (warmup_ratio를 사용)
        num_warmup_steps = int(num_training_steps * self.hparams.model_cfg.warmup_ratio)

        # 4. 코사인 스케줄러 생성
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # 5. 옵티마이저와 스케줄러를 함께 반환
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]