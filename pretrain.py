from transformers import (BertModel, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer,
    TrainingArguments)
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import sys
import os
import yaml
from dataset.dataset_pretrain import *

"""train-validation split"""
def split(file_path, seed):
    dataset = pd.read_csv(file_path).values
    train_data, valid_data = train_test_split(dataset, test_size=0.2, random_state=seed)
    return train_data, valid_data

def main(config):

    """Use BERT configuration"""
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

    """Construct MLM model"""
    model = BertForMaskedLM(config=bert_config)
    
    tokenizer = BertTokenizer.from_pretrained(config["dataset"]["vocab_path"], do_lower_case=False)

    """Load Data"""
    train_data, valid_data = split(config['file_path'], config['dataset']['seed'])
    data_train = PretrainDataset(dataset=train_data, tokenizer=tokenizer, blocksize=config["dataset"]['blocksize'])
    data_valid = PretrainDataset(dataset=valid_data, tokenizer=tokenizer, blocksize=config["dataset"]['blocksize'])

    """Set DataCollator"""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config['mlm_probability']
    )


    """Training arguments"""
    training_args = TrainingArguments(
        output_dir=config['save_path'],
        overwrite_output_dir=config['overwrite_output_dir'],
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['dataset']['batch_size'],
        per_device_eval_batch_size=config['dataset']['batch_size'],
        save_strategy=config['save_strategy'],
        save_total_limit=config['save_total_limit'],
        fp16=config['fp16'],
        logging_strategy=config['logging_strategy'],
        evaluation_strategy=config['evaluation_strategy'],
        learning_rate=config['lr_rate'],
        lr_scheduler_type=config['scheduler_type'],
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        report_to=config['report_to'],
        dataloader_num_workers=config['dataset']['dataloader_num_workers'],
        ddp_backend=config['ddp_backend'],
        fsdp=config['fsdp'],
        load_best_model_at_end=True,
    )

    """Set Trainer"""
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data_train,
        eval_dataset=data_valid
    )

    """Train and save model"""
    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=config['load_checkpoint'])
    trainer.save_model(config["save_path"])

if __name__ == "__main__":

    config = yaml.load(open("config_pretrain.yaml", "r"), Loader=yaml.FullLoader)

    """Run the main function"""
    main(config)