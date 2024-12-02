import numpy as np
import os
import sys
import yaml

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm

from transformers import BertModel, BertConfig

class Cloud(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        bert_config = BertConfig(
            hidden_size=config['num_attention_heads'] * 64,
            intermediate_size=config['num_attention_heads'] * 64 * 4,
            max_position_embeddings=config['max_position_embeddings'],
            num_attention_heads=config['num_attention_heads'],
            num_hidden_layers=config['num_hidden_layers'],
            type_vocab_size=1,
            hidden_dropout_prob=config['hidden_dropout_prob'],
            attention_probs_dropout_prob=config['attention_probs_dropout_prob'],
        )

        self.bert = BertModel(config=bert_config)

        layers = []
        layers.append(nn.Dropout(config["dropout"]))
        
        # Add the desired number of layers based on the hyperparameter
        for _ in range(config["num_layers"]):
            layers.append(nn.Linear(bert_config.hidden_size, bert_config.hidden_size))
            layers.append(nn.SiLU())
        
        # Add the final layer
        layers.append(nn.Linear(bert_config.hidden_size, 1))
        
        # Create the sequential model
        self.Regressor = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.Regressor(logits)
        return output