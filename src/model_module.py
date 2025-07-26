# src/model_module.py

# import pytorch_lightning as pl
# from transformers import (
#     BartForConditionalGeneration,
#     get_cosine_schedule_with_warmup,
# )
# import torch
# import evaluate # Use the Hugging Face evaluate library
# import wandb

# class SummaryModelModule(pl.LightningModule):
#     """
#     PyTorch Lightning의 모델 모듈.
#     모델 정의, 학습/검증 단계, 옵티마이저 설정을 담당합니다.
#     """

#     def __init__(self, model_cfg, tokenizer):
#         super().__init__()
#         # 설정을 저장하여 나중에 쉽게 불러올 수 있게 합니다.
#         # tokenizer는 객체이므로 저장하지 않습니다.
#         self.save_hyperparameters(ignore=["tokenizer"])
#         self.tokenizer = tokenizer

#         # ROUGE 점수 계산을 위한 메트릭 로드
#         self.rouge_metric = evaluate.load("rouge")

#         # 모델 로딩 (베이스라인의 load_tokenizer_and_model_for_train 함수 역할)
#         self.model = BartForConditionalGeneration.from_pretrained(
#             self.hparams.model_cfg.pretrained_model_name_or_path
#         )
#         # DataModule에서 special token이 추가된 tokenizer의 길이에 맞게
#         # 모델의 임베딩 크기를 조정합니다.
#         self.model.resize_token_embeddings(len(tokenizer))
#         self.validation_step_outputs = []


# src/model_module.py

import pytorch_lightning as pl
from transformers import (
    # 'Bart...' 대신 'Auto...'를 사용하도록 변경
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup,
)
import torch
import evaluate # Use the Hugging Face evaluate library
import wandb

class SummaryModelModule(pl.LightningModule):
    """
    PyTorch Lightning의 모델 모듈.
    모델 정의, 학습/검증 단계, 옵티마이저 설정을 담당합니다.
    """

    def __init__(self, model_cfg, tokenizer):
        super().__init__()
        # 설정을 저장하여 나중에 쉽게 불러올 수 있게 합니다.
        # tokenizer는 객체이므로 저장하지 않습니다.
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = tokenizer

        # ROUGE 점수 계산을 위한 메트릭 로드
        self.rouge_metric = evaluate.load("rouge")

        # 모델 로더를 '만능 로더'인 AutoModelForSeq2SeqLM로 변경
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_cfg.pretrained_model_name_or_path
        )
        # DataModule에서 special token이 추가된 tokenizer의 길이에 맞게
        # 모델의 임베딩 크기를 조정합니다.
        self.model.resize_token_embeddings(len(tokenizer))
        self.validation_step_outputs = []
        
    def forward(self, **inputs):
        """모델의 순전파를 정의합니다."""
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        """학습 단계를 정의합니다."""
        # 모델의 forward pass를 실행하고 loss를 얻습니다.
        # BartForConditionalGeneration은 'labels'가 제공되면 자동으로 loss를 계산합니다.
        outputs = self(**batch)
        loss = outputs.loss

        # 학습률 로깅
        lr = self.optimizers().param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # 학습 손실을 'train_loss'라는 이름으로 로깅합니다.
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """검증 단계를 정의합니다."""
        # 모델을 사용하여 요약문 생성
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.hparams.model_cfg.generate_max_length,
            num_beams=self.hparams.model_cfg.num_beams,
        )

        # 생성된 토큰과 정답 토큰을 텍스트로 디코딩
        decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        # labels에서 padding token은 무시
        labels = torch.where(batch['labels'] != -100, batch['labels'], self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ROUGE 점수 계산을 위해 예측과 정답을 추가
        self.rouge_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        # 시각화를 위해 첫 번째 배치만 저장
        if batch_idx == 0:
            original_dialogue = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            self.validation_step_outputs.append({
                "dialogue": original_dialogue,
                "summary": decoded_labels,
                "generated": decoded_preds
            })

    def on_validation_epoch_end(self):
        """검증 에포크가 끝날 때 호출됩니다."""
        # ROUGE 점수 계산 및 로깅
        rouge_scores = self.rouge_metric.compute()
        self.log_dict(rouge_scores, prog_bar=True, logger=True)

        # 생성된 텍스트 샘플을 wandb.Table로 만들어 로깅
        outputs = self.validation_step_outputs[0]
        table = wandb.Table(columns=["Dialogue", "Ground Truth", "Generated"])
        for i in range(len(outputs["dialogue"])):
            table.add_data(outputs["dialogue"][i], outputs["summary"][i], outputs["generated"][i])

        self.logger.experiment.log({"Validation Samples": table})
        self.validation_step_outputs.clear() # 다음 에포크를 위해 리스트 비우기

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """추론 단계를 정의합니다."""
        # `trainer.predict()`가 호출될 때 실행됩니다.
        generated_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.hparams.model_cfg.generate_max_length,
            num_beams=self.hparams.model_cfg.num_beams,
        )
        
        # 생성된 요약문을 텍스트로 디코딩하여 반환합니다.
        decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded_preds

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