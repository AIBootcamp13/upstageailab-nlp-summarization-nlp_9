# src/data_module.py

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
        # 설정 파일들을 인스턴스 변수로 저장
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.batch_size = data_cfg.batch_size

        # 모델에 맞는 토크나이저 불러오기
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.pretrained_model_name_or_path
        )
        
    def setup(self, stage: str = None):
        # 훈련/검증 단계일 때
        if stage == 'fit' or stage is None:
            # 1. 훈련/검증 데이터 로드
            train_df = pd.read_csv(self.data_cfg.train_path)
            val_df = pd.read_csv(self.data_cfg.val_path)

            # 2. 토크나이징
            print("--- [DataModule] 'input_text' 컬럼을 사용하여 데이터를 토크나이징합니다. ---")
            tokenized_train = self.tokenizer(
                train_df['input_text'].tolist(),
                text_target=train_df['english_summary'].tolist(),
                padding='longest', truncation=True, return_tensors="pt",
                max_length=self.model_cfg.encoder_max_len,
                # 요약문(타겟)의 최대 길이도 별도로 지정해줄 수 있음
                # max_target_length=self.model_cfg.decoder_max_len 
            )
            tokenized_val = self.tokenizer(
                val_df['input_text'].tolist(),
                text_target=val_df['english_summary'].tolist(),
                padding='longest', truncation=True, return_tensors="pt",
                max_length=self.model_cfg.encoder_max_len,
                # max_target_length=self.model_cfg.decoder_max_len
            )

            self.train_dataset = SummaryDataset(tokenized_train)
            self.val_dataset = SummaryDataset(tokenized_val)

        # 테스트 단계일 때
        if stage == 'test' or stage is None:
            # 여기서는 편의상 검증 데이터를 테스트 데이터로도 사용
            test_df = pd.read_csv(self.data_cfg.val_path)
            
            print("--- [DataModule] 테스트 데이터를 토크나이징합니다. ---")
            tokenized_test = self.tokenizer(
                test_df['input_text'].tolist(),
                text_target=test_df['english_summary'].tolist(),
                padding='longest', truncation=True, return_tensors="pt",
                max_length=self.model_cfg.encoder_max_len,
                # max_target_length=self.model_cfg.decoder_max_len
            )
            self.test_dataset = SummaryDataset(tokenized_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        # setup에서 정의된 test_dataset을 사용
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)