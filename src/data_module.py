# src/data_module.py (최종 안정화 버전)
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

class SummaryDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.tokenized_data.items()}

    def __len__(self):
        return len(self.tokenized_data['input_ids'])

class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg, model_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.batch_size = data_cfg.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.pretrained_model_name_or_path
        )

    def setup(self, stage: str = None):
        # 훈련/검증 단계
        if stage == 'fit' or stage is None:
            train_df = pd.read_csv(self.data_cfg.train_path)
            val_df = pd.read_csv(self.data_cfg.val_path)
            
            # --- 라이브러리 버전에 상관없이 가장 안정적인 토크나이징 방식 ---
            # 1. 입력(input_text) 먼저 토크나이징
            tokenized_train = self.tokenizer(
                train_df['input_text'].tolist(), padding='longest', truncation=True,
                return_tensors="pt", max_length=self.model_cfg.encoder_max_len,
            )
            # 2. 정답(english_summary)을 별도로 토크나이징해서 'labels' 생성
            with self.tokenizer.as_target_tokenizer():
                labels_train = self.tokenizer(
                    train_df['english_summary'].tolist(), padding='longest', truncation=True,
                    return_tensors="pt", max_length=self.model_cfg.decoder_max_len,
                )
            # 3. 입력 결과에 'labels'를 수동으로 추가
            tokenized_train['labels'] = labels_train['input_ids']

            # 검증 데이터도 동일하게 처리
            tokenized_val = self.tokenizer(
                val_df['input_text'].tolist(), padding='longest', truncation=True,
                return_tensors="pt", max_length=self.model_cfg.encoder_max_len,
            )
            with self.tokenizer.as_target_tokenizer():
                labels_val = self.tokenizer(
                    val_df['english_summary'].tolist(), padding='longest', truncation=True,
                    return_tensors="pt", max_length=self.model_cfg.decoder_max_len,
                )
            tokenized_val['labels'] = labels_val['input_ids']
            
            self.train_dataset = SummaryDataset(tokenized_train)
            self.val_dataset = SummaryDataset(tokenized_val)

        # 예측 단계 (이 부분은 이전의 개선된 버전 그대로 유지)
        if stage == 'predict' or stage is None:
            if hasattr(self, 'predict_df') and isinstance(self.predict_df, pd.DataFrame):
                print(f"--- [DataModule] Predict stage: 제공된 DataFrame을 토크나이징합니다. ---")
                tokenized_predict = self.tokenizer(
                    self.predict_df['input_text'].tolist(),
                    padding='longest', truncation=True, return_tensors="pt",
                    max_length=self.model_cfg.encoder_max_len,
                )
                self.predict_dataset = SummaryDataset(tokenized_predict)
            else:
                predict_file_path = self.data_cfg.get("predict_path", self.data_cfg.val_path)
                predict_df = pd.read_csv(predict_file_path)
                if 'input_text' not in predict_df.columns:
                     predict_df['topic'] = 'unknown'
                     predict_df['input_text'] = predict_df.apply(lambda row: f"summarize: topic: {row['topic']} dialogue: {row['dialogue']}", axis=1)
                tokenized_predict = self.tokenizer(predict_df['input_text'].tolist(), padding='longest', truncation=True, return_tensors="pt", max_length=self.model_cfg.encoder_max_len)
                self.predict_dataset = SummaryDataset(tokenized_predict)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        # predict_dataset이 setup에서 정의되었는지 확인
        if not hasattr(self, 'predict_dataset'):
            self.setup('predict')
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=4)

    def get_few_shot_example(self):
        train_df = pd.read_csv(self.data_cfg.train_path)
        example_row = train_df.sample(1).iloc[0]
        return example_row['input_text'], example_row['english_summary']