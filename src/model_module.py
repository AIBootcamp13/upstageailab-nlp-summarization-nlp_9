# src/model_module.py (최종 개선 버전)
import pytorch_lightning as pl
from transformers import (
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup,
)
# PEFT 라이브러리 import
from peft import get_peft_model, LoraConfig, TaskType
import torch
import evaluate

class SummaryModelModule(pl.LightningModule): # PyTorch Lightning의 LightningModule을 상속받아 모델 훈련, 검증, 예측 등을 관리합니다.
    """
    LoRA 설정을 YAML에서 불러오고, 안정성을 개선한 최종 모델 모듈.
    """
    def __init__(self, model_cfg, tokenizer):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"]) # 왜 tokenizer를 제외해? 답: tokenizer는 모델의 하이퍼파라미터가 아니기 때문 / tokenizer를 제외한 나머지 인자들만 저장하겠다는 뜻
        self.model_cfg = self.hparams.model_cfg # 코드를 간결하게 하기 위해 변수에 저장
        self.tokenizer = tokenizer
        
        # 1. 기본 모델 로드
        base_model = AutoModelForSeq2SeqLM.from_pretrained( # AutoModelForSeq2SeqLM는 시퀀스-투-시퀀스 모델을 위한 기본 클래스입니다.
            self.model_cfg.pretrained_model_name_or_path
        )

        # 2. (개선) special_tokens가 있다면, PEFT 적용 전에 embedding 리사이즈
        if self.model_cfg.get("special_tokens"):
            base_model.resize_token_embeddings(len(tokenizer))

        # 3. (개선) 메모리 절약을 위한 gradient checkpointing 활성화
        base_model.gradient_checkpointing_enable()

        # 4. (개선) LoRA 설정을 YAML 파일에서 명시적으로 불러와 적용
        if self.model_cfg.get("lora_config"):
            print("✅ [PEFT] YAML 파일에서 LoRA 설정을 불러와 적용합니다...")
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=self.model_cfg.lora_config.r,
                lora_alpha=self.model_cfg.lora_config.lora_alpha,
                lora_dropout=self.model_cfg.lora_config.lora_dropout,
                target_modules=self.model_cfg.lora_config.target_modules,
                bias="none", # 보통 "none" 또는 "all"로 설정
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters() # 훈련 가능한 파라미터 수 출력
        else:
            print("ℹ️ [PEFT] LoRA 설정이 없습니다. Full-finetuning을 진행합니다.")
            self.model = base_model
            
        self.rouge_metric = evaluate.load("rouge")
        self.validation_step_outputs = []

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # 5. (개선) 'ssss' 버그 재발을 막기 위해 decoder_start_token_id 삭제
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=self.model_cfg.generate_max_length,
            num_beams=self.model_cfg.num_beams,
        )
        
        labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        generated_summaries = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.rouge_metric.add_batch(predictions=generated_summaries, references=labels)
        
        # 6. (개선) 5 에폭마다 테이블을 기록하여 WandB 자원 절약
        #    첫 에폭(epoch 0)에는 무조건 기록하여 초기 성능 확인
        if self.current_epoch == 0 or (self.current_epoch + 1) % 5 == 0:
            if batch_idx == 0:
                self.validation_step_outputs.append({
                    "dialogue_ids": batch["input_ids"], "label_ids": batch["labels"], "generated_ids": generated_ids
                })

    def on_validation_epoch_end(self): # 검증 에폭이 끝날 때 호출되는 메소드
        results = self.rouge_metric.compute()
        self.log_dict(results, on_epoch=True, prog_bar=True)

        if self.validation_step_outputs:
            first_batch_outputs = self.validation_step_outputs[0]
            dialogues = self.tokenizer.batch_decode(first_batch_outputs["dialogue_ids"], skip_special_tokens=True)
            ground_truths = self.tokenizer.batch_decode(first_batch_outputs["label_ids"], skip_special_tokens=True)
            generateds = self.tokenizer.batch_decode(first_batch_outputs["generated_ids"], skip_special_tokens=True)
            columns = ["Dialogue", "Ground Truth", "Generated"]
            data = [[dialogue, truth, gen] for dialogue, truth, gen in zip(dialogues, ground_truths, generateds)]
            self.logger.log_table(key=f"Validation Samples @ Epoch {self.current_epoch}", columns=columns, data=data)

        self.validation_step_outputs.clear() # 클리어 하는 이유: 다음 에폭에서 새로운 결과를 저장하기 위해


    # 동적 길이로 하니까 요약문이 잘리는 현상이 발생함 -> eda 결과 안전한 고정 길이(178)로 변경
    # predict_step 메소드 전체를 아래 코드로 교체
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # --- 최종 수정: 데이터 기반의 안전한 고정 길이(178) 사용 ---
        
        # 기본 생성 옵션 설정
        gen_kwargs = {
            "max_length": 178, # <-- 데이터 분석으로 찾은 최적의 길이로 고정!
            "num_beams": self.model_cfg.get("num_beams", 4),
            "no_repeat_ngram_size": self.model_cfg.get("no_repeat_ngram_size", 2),
            "length_penalty": self.model_cfg.get("length_penalty", 1.0),
            "repetition_penalty": self.model_cfg.get("repetition_penalty", 1.0),
            "temperature": self.model_cfg.get("temperature", 1.0),
            "top_p": self.model_cfg.get("top_p", 1.0),
        }
        
        # inference.py에서 커맨드 라인으로 추가 옵션을 주면, 여기서 덮어쓰기 됨
        if "generation" in self.hparams and self.hparams.generation:
            override_kwargs = {k: v for k, v in self.hparams.generation.items() if v is not None}
            gen_kwargs.update(override_kwargs)
        
        # 이제 max_length는 고정되었으므로, print문에서 제거해도 됨
        print(f"  [Generating] num_beams: {gen_kwargs['num_beams']}")
        
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            **gen_kwargs
        )
        decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded_preds

    # --- 개선점: 최적화기와 스케줄러 설정 ---
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.model_cfg.learning_rate,
            weight_decay=self.model_cfg.weight_decay,
        )
        
        # --- 개선점: 스케줄러를 CosineAnnealingWarmRestarts로 업그레이드 ---
        num_training_steps = self.trainer.estimated_stepping_batches
        
        # T_0: 첫 번째 재시작까지의 스텝 수 (예: 2 에포크)
        # 1 에포크당 스텝 수 계산이 필요
        steps_per_epoch = num_training_steps // self.trainer.max_epochs
        t_0 = 2 * steps_per_epoch

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t_0,      # 2 에포크마다 학습률 재시작
            T_mult=1,   # 재시작 주기 변경 없음
            eta_min=self.model_cfg.learning_rate * 0.1 # 최소 학습률
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]