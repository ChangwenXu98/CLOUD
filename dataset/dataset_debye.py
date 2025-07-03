import json
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from matbench.bench import MatbenchBenchmark
from transformers import BertTokenizer
from sklearn.preprocessing import StandardScaler
import functools

class CloudDataset(Dataset):
    def __init__(self, df, tokenizer, blocksize, train=True):
        """
        df: input datafram
        tokenizer: used for tokenization
        blocksize: the max number of tokens
        train: train or validation mode
        """
        self.df = df
        self.tokenizer = tokenizer
        self.blocksize = blocksize
        self.is_train = train
    
        # self.df = self.df.iloc[:5, :]
        print("Number of data:", self.df.shape[0])

    def __len__(self):
        self.len = len(self.df)
        return self.len

    # Cache data for faster training
    @functools.lru_cache(maxsize=None)  # noqa: B019
    def __getitem__(self, index):
        gen_str = self.df.iloc[index, 0]
        target = self.df.iloc[index, 1]
        n_atoms = self.df.iloc[index, 2]

        encoding = self.tokenizer(
            gen_str,
            add_special_tokens=True,
            max_length=self.blocksize,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            target=target, 
            n_atoms=n_atoms
        )
    
class CloudDatasetWrapper(object):
    def __init__(
        self, dataset_name, batch_size, vocab_path, blocksize, num_workers, seed, valid_size
    ):
        super(object, self).__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=False)
        self.blocksize = blocksize
        self.seed = seed
        self.valid_size = valid_size

    def load_dataset(self, dataset_name, fold, is_train, val=False):
        if is_train:   
            df = pd.read_csv(os.path.join("data/insight", f"{dataset_name}_train_and_val.csv"))
        else:
            df = pd.read_csv(os.path.join("data/insight", f"{dataset_name}_test.csv"))
                
        return df

    def get_data_loaders(self, fold):
        self.df_train = self.load_dataset(self.dataset_name, fold, is_train=True)
        self.df_test = self.load_dataset(self.dataset_name, fold, is_train=False)
        num_data = self.df_train.shape[0]
        indices = list(self.df_train.index)
        random_state = np.random.RandomState(seed=self.seed)
        random_state.shuffle(indices)
        split = int(np.floor(self.valid_size * num_data))
        val_idx = indices[:split]
        train_idx = indices[split:]
        self.df_valid = self.df_train.loc[val_idx, :].reset_index(drop=True)
        self.df_train = self.df_train.loc[train_idx, :].reset_index(drop=True)
        self.mu_train = np.mean(self.df_train.values[:, 1])
        self.train_mad = np.mean(np.abs(self.df_train.values[:, 1] - self.mu_train))
        # self.mu_test = np.mean(self.df_test.values[:, 1])
        
        train_dataset = CloudDataset(self.df_train, self.tokenizer, self.blocksize)
        valid_dataset = CloudDataset(self.df_valid, self.tokenizer, self.blocksize, train=False)
        test_dataset = CloudDataset(self.df_test, self.tokenizer, self.blocksize, train=False)
        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader