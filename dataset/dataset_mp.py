import json
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
from sklearn.model_selection import train_test_split

class CloudDataset(Dataset):
    def __init__(self, dataset, tokenizer, blocksize, mu=0.0, std=1.0, train=True):
    
        self.tokenizer = tokenizer
        self.blocksize = blocksize
        self.is_train = train
        self.df = dataset

        # TODO: no normalization
        if self.is_train:
            self.mu = np.mean(self.df.values[:, 1])
            self.std = np.std(self.df.values[:, 1])
        else:
            self.mu = mu
            self.std = std
        self.df.iloc[:, 1] = (self.df.iloc[:, 1] - self.mu) / self.std
        # self.mu = mu
        # self.std = std
    
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
            target=target
        )
    
class CloudDatasetWrapper(object):
    def __init__(
        self, dataset_name, batch_size, vocab_path, blocksize, num_workers, seed, valid_size
    ):
        super(object, self).__init__()
        self.dataset = pd.read_csv(dataset_name)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=False)
        self.blocksize = blocksize
        self.seed = seed
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_data, valid_data = train_test_split(self.dataset, test_size=self.valid_size, random_state=self.seed)
        train_dataset = CloudDataset(train_data, self.tokenizer, self.blocksize)
        # TODO no normalization
        valid_dataset = CloudDataset(valid_data, self.tokenizer, self.blocksize, 
                                         train_dataset.mu, train_dataset.std, train=False)
        # valid_dataset = CloudDataset(valid_data, self.tokenizer, self.blocksize, train=False)
        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_dataset, valid_dataset, train_loader, valid_loader