import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.lr_sched import *
import shutil
from utils.LLRD import roberta_base_AdamW_LLRD
from model.cloud import Cloud
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class load_dataset(Dataset):
     def __init__(self, dataset, vocab_path, blocksize):

        self.tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=False)
        self.blocksize = blocksize
        self.df = dataset

     def __len__(self):
        self.len = len(self.df)
        return self.len

     def __getitem__(self, index):
        gen_str = self.df.iloc[index, 0]

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
        )
                

if __name__ == "__main__":
    config = yaml.load(open("config_wbm.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    if torch.cuda.is_available() and config['gpu'] != 'cpu':
        device = config['gpu']
    else:
        device = 'cpu'
    print("Running on:", device)

    model = Cloud(config["model"])
    checkpoints_folder = os.path.join('runs', config['load_model'], 'checkpoints')
    state_dict = torch.load(os.path.join(checkpoints_folder, 'model_best.pth'), map_location=torch.device(device))
    model.load_state_dict(state_dict["model_state_dict"])
    mu = state_dict["mu"]
    std = state_dict["std"]
    print(f"mu: {mu}")
    print(f"std: {std}")
    model.to(device)

    df = pd.read_csv(config["dataset"]["data_path"])
    dataset = load_dataset(df, config["dataset"]["vocab_path"], config["dataset"]["blocksize"])
    dataloader = DataLoader(dataset, config["dataset"]["batch_size"], shuffle=False, num_workers=config["dataset"]["num_workers"])

    prediction = []
    with torch.no_grad():
        model.eval()
        for bn, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device) 
            output = model(input_ids, attention_mask)
            output = output * std + mu
            prediction.append(output.detach().cpu().numpy())
    
    prediction = np.concatenate(prediction, axis=0)

    df_pred = pd.DataFrame(prediction, columns=["eform_pred"])
    df_pred.to_csv(config["save_path"], index=False)