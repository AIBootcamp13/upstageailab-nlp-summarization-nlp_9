# src/data_module.py

import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

# 전처리 함수 (어제 추가했던 것)
def clean_text(text):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text

# 클래스 설명: 요약 데이터셋
# 이 클래스는 Hugging Face의 Dataset을 상속받아, 요약 모델 학습에 필요한 데이터셋을 정의합니다.
# 데이터는 'input_text'와 'english_summary' 컬럼을 포함하며,
# 이를 토큰화하여 모델이 이해할 수 있는 형태로 변환합니다.
# 또한, T5 모델의 경우 topic_token을 접두사로 사용하여 입력을 생성합니다.
# 이 클래스는 PyTorch의 Dataset 인터페이스를 구현하여, DataLoader와 함께 사용될 수 있습니다.
class SummaryDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_data.items()}
        return item

    def __len__(self):
        return len(self.tokenized_data['input_ids'])

class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg, model_cfg, train_df=None, val_df=None):
        super().__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        
        # ▼▼▼▼▼ 바로 이 한 줄이 빠져 있었어! ▼▼▼▼▼
        self.batch_size = data_cfg.batch_size
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 외부에서 데이터프레임을 직접 주입받을 수 있도록 함
        self.train_df_external = train_df
        self.val_df_external = val_df

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.pretrained_model_name_or_path
        )
        
        # data_cfg에 topic 정보가 있으면 special token으로 추가
        if self.data_cfg.get("topics"):
            special_tokens_to_add = list(self.data_cfg.topics)
            self.tokenizer.add_special_tokens(
                {'additional_special_tokens': special_tokens_to_add}
            )

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            if self.train_df_external is not None:
                train_df = self.train_df_external
                val_df = self.val_df_external
            else:
                train_df = pd.read_csv(self.data_cfg.train_path)
                val_df = pd.read_csv(self.data_cfg.val_path)

            is_t5 = "t5" in self.model_cfg.pretrained_model_name_or_path.lower()
            
            # ▼▼▼▼▼ 핵심 수정 부분 ▼▼▼▼▼
            # 설정 파일의 use_topic_prefix 값 (기본값 True)에 따라 분기 처리
            if is_t5 and self.data_cfg.get("use_topic_prefix", True):
                print("--- [DataModule] Topic 힌트를 사용하여 입력을 구성합니다. ---")
                train_df['input_text'] = train_df['topic_token'] + ' ' + train_df['english_dialogue']
                val_df['input_text'] = val_df['topic_token'] + ' ' + val_df['english_dialogue']
            else:
                print("--- [DataModule] Topic 힌트 없이 입력을 구성합니다. ---")
                train_df['input_text'] = train_df['english_dialogue']
                val_df['input_text'] = val_df['english_dialogue']
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            tokenized_train = self.tokenizer(
                train_df['input_text'].tolist(),
                text_target=train_df['english_summary'].tolist(),
                padding=True, truncation=True, return_tensors="pt",
                max_length=self.model_cfg.encoder_max_len
            )
            tokenized_val = self.tokenizer(
                val_df['input_text'].tolist(),
                text_target=val_df['english_summary'].tolist(),
                padding=True, truncation=True, return_tensors="pt",
                max_length=self.model_cfg.encoder_max_len
            )
            
            self.train_dataset = SummaryDataset(tokenized_train)
            self.val_dataset = SummaryDataset(tokenized_val)
    def _tokenize_data(self, encoder_input, decoder_input=None, decoder_output=None):
        tokenized_encoder = self.tokenizer(encoder_input, return_tensors="pt", padding=True, truncation=True, max_length=self.model_cfg.encoder_max_len)
        if decoder_input is not None and decoder_output is not None:
            tokenized_decoder = self.tokenizer(decoder_input, return_tensors="pt", padding=True, truncation=True, max_length=self.model_cfg.decoder_max_len)
            tokenized_labels = self.tokenizer(decoder_output, return_tensors="pt", padding=True, truncation=True, max_length=self.model_cfg.decoder_max_len)
            return {'input_ids': tokenized_encoder.input_ids, 'attention_mask': tokenized_encoder.attention_mask, 'decoder_input_ids': tokenized_decoder.input_ids, 'decoder_attention_mask': tokenized_decoder.attention_mask, 'labels': tokenized_labels.input_ids}
        else:
            return {'input_ids': tokenized_encoder.input_ids, 'attention_mask': tokenized_encoder.attention_mask}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)