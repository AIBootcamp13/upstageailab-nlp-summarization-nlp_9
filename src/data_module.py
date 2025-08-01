# # src/data_module.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

class SummaryDataset(Dataset):
    """
    토큰화된 데이터를 받아 PyTorch Dataset으로 변환하는 클래스.
    """
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __getitem__(self, idx):
        # 각 아이템을 딕셔너리 형태로 반환
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_data.items()}
        return item

    def __len__(self):
        return len(self.tokenized_data['input_ids'])

class SummaryDataModule(pl.LightningDataModule):
    """
    데이터 로드, 전처리, DataLoader 생성을 총괄하는 PyTorch Lightning의 DataModule.
    """
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
            tokenized_train = self.tokenizer(
                train_df['input_text'].tolist(), text_target=train_df['english_summary'].tolist(),
                padding='longest', truncation=True, return_tensors="pt", max_length=self.model_cfg.encoder_max_len
            )
            tokenized_val = self.tokenizer(
                val_df['input_text'].tolist(), text_target=val_df['english_summary'].tolist(),
                padding='longest', truncation=True, return_tensors="pt", max_length=self.model_cfg.encoder_max_len
            )
            self.train_dataset = SummaryDataset(tokenized_train)
            self.val_dataset = SummaryDataset(tokenized_val)

        # 테스트 또는 예측 단계
        if stage == 'test' or stage == 'predict' or stage is None:
            test_df_path = os.path.join(os.getcwd(), 'data', 'raw', 'test.csv')
            test_df = pd.read_csv(test_df_path)
            test_df['input_text'] = "summarize: dialogue: " + test_df['dialogue']
            tokenized_test = self.tokenizer(
                test_df['input_text'].tolist(),
                padding='longest', truncation=True, return_tensors="pt", max_length=self.model_cfg.encoder_max_len
            )
            if 'labels' in tokenized_test:
                del tokenized_test['labels']
            # test_dataset과 predict_dataset을 모두 정의
            self.test_dataset = SummaryDataset(tokenized_test)
            self.predict_dataset = SummaryDataset(tokenized_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
        
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=4)