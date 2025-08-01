# src/model_module.py

import pytorch_lightning as pl
from transformers import (
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup,
)
import torch
import evaluate
import wandb

import pytorch_lightning as pl
from transformers import (
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup,
)
import torch
import evaluate
import wandb
# LoRA를 위한 import 추가
from peft import get_peft_model, LoraConfig, TaskType

class SummaryModelModule(pl.LightningModule):
    def __init__(self, model_cfg, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer
        self.rouge_metric = evaluate.load("rouge")

        # 1. 기본 모델 불러오기
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_cfg.pretrained_model_name_or_path
        )
        self.model.gradient_checkpointing_enable()  # ✅ 메모리 절약을 위한 gradient checkpointing 활성화. 로라 전에 적용하는게 안전함

        # 2. LoRA 설정 정의
        # 대회 가이드에 나온 추천값을 기반으로 설정
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q", "v"], # T5 모델의 어텐션 레이어에 적용
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        # 3. 모델에 LoRA 적용
        self.model = get_peft_model(self.model, lora_config)
        
        # LoRA 적용 후 훈련 가능한 파라미터 수 출력
        self.model.print_trainable_parameters()
        
        self.model.resize_token_embeddings(len(tokenizer))
        self.validation_step_outputs = []
        
    # forward, training_step 등 나머지 메소드는 기존과 동일하게 유지
    # (이 아래로는 수정할 필요 없음)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # ... (기존 코드와 동일)
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
                "dialogue_ids": batch["input_ids"], "label_ids": batch["labels"], "generated_ids": generated_ids
            })


    def on_validation_epoch_end(self):
        results = self.rouge_metric.compute()
        self.log_dict(results, on_epoch=True, prog_bar=True)

        if self.validation_step_outputs:
            # 5 에폭마다 한 번씩만 테이블을 기록하도록 변경
            if self.current_epoch % 5 == 0:
                first_batch_outputs = self.validation_step_outputs[0]
                dialogues = self.tokenizer.batch_decode(first_batch_outputs["dialogue_ids"], skip_special_tokens=True)
                ground_truths = self.tokenizer.batch_decode(first_batch_outputs["label_ids"], skip_special_tokens=True)
                generateds = self.tokenizer.batch_decode(first_batch_outputs["generated_ids"], skip_special_tokens=True)
                columns = ["Dialogue", "Ground Truth", "Generated"]
                data = [[dialogue, truth, gen] for dialogue, truth, gen in zip(dialogues, ground_truths, generateds)]
                self.logger.log_table(key="Validation Samples", columns=columns, data=data)

        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # ... (기존 코드와 동일)
        gen_kwargs = {
            "max_length": self.hparams.model_cfg.generate_max_length,
            "num_beams": self.hparams.model_cfg.num_beams,
            "no_repeat_ngram_size": self.hparams.model_cfg.get("no_repeat_ngram_size", 2),
            "decoder_start_token_id": self.model.config.pad_token_id,
        }
        if "generation" in self.hparams and self.hparams.generation:
            override_kwargs = {k: v for k, v in self.hparams.generation.items() if v is not None}
            gen_kwargs.update(override_kwargs)
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], **gen_kwargs
        )
        decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded_preds


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