# src/data_module.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

class SummaryDataset(Dataset):
    """
    HuggingFace의 Bart 모델을 위한 커스텀 데이터셋 클래스.
    PyTorch의 Dataset 클래스를 상속받아 만듭니다.
    """
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __getitem__(self, idx):
        # 각 키에 대해 해당 인덱스의 텐서를 가져옵니다.
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_data.items()}
        return item

    def __len__(self):
        # 데이터셋의 길이는 input_ids의 길이로 결정됩니다.
        return len(self.tokenized_data['input_ids'])


class SummaryDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning의 데이터 모듈.
    데이터 로딩, 전처리, 데이터로더 생성을 모두 담당합니다.
    """
    def __init__(self, data_cfg, model_cfg):
        super().__init__()
        self.data_path = data_cfg.path
        self.batch_size = data_cfg.batch_size

        # 설정 파일에서 모델 및 토크나이저 설정을 가져옵니다.
        self.model_cfg = model_cfg

        # save_hyperparameters()를 통해 설정값을 저장하여 나중에 쉽게 불러올 수 있습니다.
        self.save_hyperparameters()

        # 토크나이저는 한 번만 초기화합니다.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.pretrained_model_name_or_path
        )
        # 설정 파일에 정의된 스페셜 토큰을 추가합니다.
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': list(self.model_cfg.special_tokens)}
        )

    def setup(self, stage: str = None):
        """
        데이터를 로드하고 전처리하는 메인 메소드.
        'fit' 단계에서 train/validation 데이터를 준비합니다.
        """
        if stage == 'fit' or stage is None:
            train_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
            val_df = pd.read_csv(os.path.join(self.data_path, 'dev.csv'))

            # 베이스라인의 Preprocess 클래스 로직을 여기에 통합합니다.
            # 1. 입력/출력 텍스트 생성
            train_encoder_input = train_df['dialogue'].tolist()
            train_decoder_input = [self.model_cfg.bos_token + str(s) for s in train_df['summary']]
            train_decoder_output = [str(s) + self.model_cfg.eos_token for s in train_df['summary']]

            val_encoder_input = val_df['dialogue'].tolist()
            val_decoder_input = [self.model_cfg.bos_token + str(s) for s in val_df['summary']]
            val_decoder_output = [str(s) + self.model_cfg.eos_token for s in val_df['summary']]

            # 2. 텍스트를 토큰화
            tokenized_train_inputs = self._tokenize_data(train_encoder_input, train_decoder_input, train_decoder_output)
            tokenized_val_inputs = self._tokenize_data(val_encoder_input, val_decoder_input, val_decoder_output)

            # 3. PyTorch Dataset 객체 생성
            self.train_dataset = SummaryDataset(tokenized_train_inputs)
            self.val_dataset = SummaryDataset(tokenized_val_inputs)

        if stage == 'test' or stage == 'predict' or stage is None:
            test_df = pd.read_csv(os.path.join(self.data_path, 'test.csv'))
            test_encoder_input = test_df['dialogue'].tolist()
            # 테스트 데이터는 정답(summary)이 없으므로 encoder_input만 토큰화합니다.
            tokenized_test_inputs = self._tokenize_data(test_encoder_input)
            self.test_dataset = SummaryDataset(tokenized_test_inputs)

    def _tokenize_data(self, encoder_input, decoder_input=None, decoder_output=None):
        """토크나이저를 사용하여 데이터를 토큰화하고 텐서로 변환하는 헬퍼 함수"""
        tokenized_encoder = self.tokenizer(
            encoder_input, return_tensors="pt", padding=True, truncation=True,
            max_length=self.model_cfg.encoder_max_len
        )

        # 학습/검증 시에만 decoder와 label 데이터를 처리합니다.
        if decoder_input is not None and decoder_output is not None:
            tokenized_decoder = self.tokenizer(
                decoder_input, return_tensors="pt", padding=True, truncation=True,
                max_length=self.model_cfg.decoder_max_len
            )

            tokenized_labels = self.tokenizer(
                decoder_output, return_tensors="pt", padding=True, truncation=True,
                max_length=self.model_cfg.decoder_max_len
            )

            return {
                'input_ids': tokenized_encoder.input_ids,
                'attention_mask': tokenized_encoder.attention_mask,
                'decoder_input_ids': tokenized_decoder.input_ids,
                'decoder_attention_mask': tokenized_decoder.attention_mask,
                'labels': tokenized_labels.input_ids
            }
        # 테스트 시에는 input_ids와 attention_mask만 반환합니다.
        else:
            return {
                'input_ids': tokenized_encoder.input_ids,
                'attention_mask': tokenized_encoder.attention_mask,
            }

    def train_dataloader(self):
        """학습용 데이터로더를 반환합니다."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        """검증용 데이터로더를 반환합니다."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        """테스트용 데이터로더를 반환합니다."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)

    def predict_dataloader(self):
        """추론용 데이터로더를 반환합니다."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)