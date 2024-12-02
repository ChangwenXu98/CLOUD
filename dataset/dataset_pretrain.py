import json
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from matbench.bench import MatbenchBenchmark
from transformers import BertTokenizer

class PretrainDataset(Dataset):
    def __init__(self, dataset, tokenizer, blocksize):
        self.tokenizer = tokenizer
        self.blocksize = blocksize
        self.dataset = dataset

        # TODO Debug
        # self.dataset = self.dataset[:5, :]

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, i):
        seq = self.dataset[i, 0]

        encoding = self.tokenizer(
            str(seq),
            add_special_tokens=True,
            max_length=self.blocksize,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # TODO Debug
        # if i == 1:
        #     print("example:", self.tokenizer.convert_ids_to_tokens(encoding["input_ids"].flatten()))

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
        )
    
# class PretrainDatasetWrapper(object):
#     def __init__(
#         self, file_path, batch_size, valid_size, vocab_path, blocksize, num_workers, seed
#     ):
#         super(object, self).__init__()
#         self.file_path = file_path
#         self.batch_size = batch_size
#         self.tokenizer = BertTokenizer(vocab_path, do_lower_case=False)
#         self.blocksize = blocksize
#         self.num_workers = num_workers
#         self.valid_size = valid_size
#         self.seed = seed

#     def get_data_loaders(self):
#         dataset = PretrainDataset(self.file_path, self.tokenizer, self.blocksize)
#         num_data = len(dataset)
#         indices = list(range(num_data))
        
#         random_state = np.random.RandomState(seed=self.seed)
#         random_state.shuffle(indices)
#         # np.random.shuffle(indices)

#         split = int(np.floor(self.valid_size * num_data))
#         valid_idx, train_idx = indices[:split], indices[split:]

#         # define samplers for obtaining training and validation batches
#         train_sampler = SubsetRandomSampler(train_idx)
#         valid_sampler = SubsetRandomSampler(valid_idx)

#         train_loader = DataLoader(
#             dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers
#         )
#         valid_loader = DataLoader(
#             dataset, batch_size=self.batch_size, sampler=valid_sampler, num_workers=self.num_workers
#         )
#         return train_loader, valid_loader