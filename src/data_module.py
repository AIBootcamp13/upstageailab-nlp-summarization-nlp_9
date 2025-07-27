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

class SummaryDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_data.items()}
        return item

    def __len__(self):
        return len(self.tokenized_data['input_ids'])

class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.data_path = data_cfg.path
        self.batch_size = data_cfg.batch_size
        self.save_hyperparameters()

        self.is_t5 = "t5" in self.model_cfg.pretrained_model_name_or_path.lower()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.pretrained_model_name_or_path
        )
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': list(self.model_cfg.special_tokens)}
        )

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            train_df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
            val_df = pd.read_csv(os.path.join(self.data_path, 'dev.csv'))
            
            # 텍스트 클리닝
            train_df['dialogue'] = train_df['dialogue'].apply(clean_text)
            train_df['summary'] = train_df['summary'].apply(clean_text)
            val_df['dialogue'] = val_df['dialogue'].apply(clean_text)
            val_df['summary'] = val_df['summary'].apply(clean_text)

            if self.is_t5:
                train_encoder_input = ["summarize: " + dialogue for dialogue in train_df['dialogue']]
                val_encoder_input = ["summarize: " + dialogue for dialogue in val_df['dialogue']]
                train_decoder_input = train_df['summary'].astype(str).tolist()
                val_decoder_input = val_df['summary'].astype(str).tolist()
            else:
                train_encoder_input = train_df['dialogue'].tolist()
                val_encoder_input = val_df['dialogue'].tolist()
                train_decoder_input = [self.model_cfg.bos_token + str(s) for s in train_df['summary']]
                val_decoder_input = [self.model_cfg.bos_token + str(s) for s in val_df['summary']]

            train_decoder_output = [str(s) + self.model_cfg.eos_token for s in train_df['summary']]
            val_decoder_output = [str(s) + self.model_cfg.eos_token for s in val_df['summary']]

            tokenized_train_inputs = self._tokenize_data(train_encoder_input, train_decoder_input, train_decoder_output)
            tokenized_val_inputs = self._tokenize_data(val_encoder_input, val_decoder_input, val_decoder_output)

            self.train_dataset = SummaryDataset(tokenized_train_inputs)
            self.val_dataset = SummaryDataset(tokenized_val_inputs)

        if stage == 'test' or stage == 'predict' or stage is None:
            test_df = pd.read_csv(os.path.join(self.data_path, 'test.csv'))
            test_df['dialogue'] = test_df['dialogue'].apply(clean_text)
            if self.is_t5:
                test_encoder_input = ["summarize: " + dialogue for dialogue in test_df['dialogue']]
            else:
                test_encoder_input = test_df['dialogue'].tolist()
            tokenized_test_inputs = self._tokenize_data(test_encoder_input)
            self.test_dataset = SummaryDataset(tokenized_test_inputs)

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