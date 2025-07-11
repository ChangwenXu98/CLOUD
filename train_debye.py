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
from dataset.dataset_debye import CloudDatasetWrapper
from model.cloud_debye import Cloud
import shutil
from utils.LLRD import roberta_base_AdamW_LLRD
import argparse
from transformers import get_cosine_schedule_with_warmup


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.log_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.dataset =CloudDatasetWrapper(**config['dataset'])
        self.criterion = nn.L1Loss()

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # set_random_seed(self.config['dataset']['seed'])

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        print("Running on:", device)

        return device
    
    @staticmethod
    def _save_config_file(ckpt_dir, config):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        # Save the updated config dictionary to a YAML file
        with open(os.path.join(ckpt_dir, 'config_debye.yaml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


    def train(self):
        loss_fold = []
        fold_num = self.config["fold_num"]
        for fold in range(fold_num-1, fold_num):
            print("Fold:", fold)
            train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = self.dataset.get_data_loaders(fold)

            model = Cloud(self.config["model"])
            
            model = self._load_weights(model)
            model = model.to(self.device)

            if type(self.config['lr']) == str: self.config['lr'] = eval(self.config['lr']) 
            if type(self.config['min_lr']) == str: self.config['min_lr'] = eval(self.config['min_lr'])
            if type(self.config['weight_decay']) == str: self.config['weight_decay'] = eval(self.config['weight_decay'])

            if self.config["optimizer"] == "adam":
                optimizer = Adam(
                    model.parameters(), self.config['lr'],
                    weight_decay=self.config['weight_decay'],
                )

            elif self.config["optimizer"] == "adamw":
                if self.config["LLRD"]:
                    optimizer = roberta_base_AdamW_LLRD(model, self.config['lr'], self.config['weight_decay'])
                else:
                    optimizer = AdamW(
                        model.parameters(), self.config['lr'],
                        weight_decay=self.config['weight_decay'],
                    )
            else:
                raise TypeError("Please choose the correct optimizer")

            ckpt_dir = os.path.join(self.writer.log_dir, 'fold_{}'.format(fold), 'checkpoints')
            self._save_config_file(ckpt_dir, self.config)

            if self.config["resume_from_checkpoint"]:
                checkpoints_folder = os.path.join('runs', self.config['resume_from_checkpoint'], 'checkpoints')
                state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
                best_valid_loss = state_dict["best_valid_loss"]
                epoch_start = state_dict["epoch"] + 1
            else:
                best_valid_loss = np.inf
                epoch_start = 0
            n_iter = 0
            valid_n_iter = 0
            for epoch_counter in range(epoch_start, self.config['epochs']):
                if epoch_counter < self.config["freeze_epoch"]:
                    print("Freeze Bert")
                    for param in model.bert.parameters():
                        param.requires_grad = False
                else:
                    print("Relax Bert")
                    for param in model.bert.parameters():
                        param.requires_grad = True
                model.train()
                for bn, batch in enumerate(tqdm(train_loader)):
                    adjust_learning_rate(optimizer, epoch_counter + bn / len(train_loader), self.config)

                    loss, _, _ = self.loss_fn(model, batch)

                    optimizer.zero_grad()
                    loss.backward()
                    # clip gradient
                    # total_norm = torch.norm(
                    #     torch.stack([torch.norm(p.grad) for p in model.parameters() if p.grad is not None])
                    # )
                    # print(f"Gradient norm: {total_norm.item()}")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.config['max_norm'])
                    optimizer.step()

                    if n_iter % self.config['log_every_n_steps'] == 0:
                        self.writer.add_scalar('loss', loss.item(), global_step=n_iter)
                        self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=n_iter)
                        print(epoch_counter, bn, 'loss', loss.item())
                    n_iter += 1

                print("Start Validation")
                valid_loss = self._validate(model, valid_loader)
                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                print('Validation', epoch_counter, 'MAE Loss', valid_loss)

                states = {
                    'model_state_dict': model.state_dict(),
                    "best_valid_loss": best_valid_loss,
                    'epoch': epoch_counter,
                    # 'mu': train_dataset.mu,
                    # 'std': train_dataset.std,
                }

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(states, os.path.join(ckpt_dir, 'model_best.pth'))

            
                torch.save(states, os.path.join(ckpt_dir, 'model.pth'))

                valid_n_iter += 1

            ckpt_best = torch.load(os.path.join(ckpt_dir, 'model_best.pth'), weights_only=False)
            model.load_state_dict(ckpt_best["model_state_dict"])

            print("Start test")
            test_loss = self._validate(model, test_loader, mae_mad=self.config["test_MAE/MAD"])
            loss_fold.append(test_loss)
            print("test loss on fold", fold, ":", test_loss)
        if self.config["test_MAE/MAD"]:
            print("Average Test MAE/MAD:", np.mean(loss_fold))
        else:
            print("Average Test MAE:", np.mean(loss_fold))
        print("Standard deviation:", np.std(loss_fold))

    def _validate(self, model, valid_loader, mae_mad=False):
        valid_loss = 0
        num_data = 0
        valid_mad = 0
        # valid_mad_2 = 0
        with torch.no_grad():
            model.eval()
            for bn, batch in enumerate(tqdm(valid_loader)):
                _, output, target = self.loss_fn(model, batch)
                # output = output * std + mu
                # target = target * std + mu
                loss = self.criterion(output.squeeze(), target.squeeze())
                valid_loss += loss.item() * output.shape[0]
                num_data += output.shape[0]
                if mae_mad:
                    valid_mad += torch.sum(torch.abs(target - self.dataset.mu_train)).item()
                    # valid_mad_2 += torch.sum(torch.abs(target - self.dataset.mu_test)).item()

            valid_loss /= num_data
            print(f"num_data: {num_data}")
            print("Valid/Test Loss: {:.4f}".format(valid_loss))
            if mae_mad:
                mae_mad_loss = valid_loss * num_data / valid_mad
                # mae_mad_loss_2 = valid_loss * num_data / valid_mad_2
                # print(f"mae_mad_loss_2: {mae_mad_loss_2}")
                return mae_mad_loss
        return valid_loss
    
    def _load_weights(self, model):
        if self.config['load_model']:
            try:
                model.bert = model.bert.from_pretrained(self.config["load_model"])
                print("Loaded pre-trained model with success.")
            except:
                print("Pre-trained weights not found. Training from scratch.")
        if self.config["resume_from_checkpoint"]:
            try:
                checkpoints_folder = os.path.join('runs', self.config['resume_from_checkpoint'], 'checkpoints')
                state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
                model.load_state_dict(state_dict['model_state_dict'])
                print("Loaded pre-trained model with success.")
            except FileNotFoundError:
                print("Pre-trained weights not found. Training from scratch.")

        return model

    def loss_fn(self, model, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device) 
        target = batch["target"].to(self.device)
        n_atoms = batch["n_atoms"].to(self.device)
        output = model(input_ids, attention_mask, T=300, n_atoms=n_atoms, n=20, mu=self.config["mu"], scale=self.config["scale"], task=self.config["task"])
        
        loss = self.criterion(output.squeeze(), target.squeeze())
        
        return loss, output, target
    
def main():

    parser = argparse.ArgumentParser(description="Get input")
    parser.add_argument("--fold_num", help="fold number", type=int)
    parser.add_argument("--lr", help="learning rate", type=float)
    parser.add_argument("--epochs", help="number of epochs", type=int)
    parser.add_argument("--warmup_epochs", help="number of warmup epochs", type=int)
    parser.add_argument("--patience_epochs", help="number of patience epochs", type=int)
    parser.add_argument("--weight_decay", help="weight_decay", type=float)
    parser.add_argument("--dataset.dataset_name", help="dataset name", type=str)
    parser.add_argument("--dataset.batch_size", help="batch size", type=int)
    parser.add_argument("--load_model", help="load model path", type=str)
    parser.add_argument("--model.num_layers", help="num of layers in MLP", type=int)
    parser.add_argument("--model.num_attention_heads", help="num of attention heads", type=int)
    parser.add_argument("--model.num_hidden_layers", help="num of hidden layers", type=int)
    parser.add_argument("--dataset.vocab_path", help="tokenizer vocab path", type=str)
    parser.add_argument("--dataset.seed", help="data random seed", type=int)
    parser.add_argument("--mu", help="mu", type=float)
    parser.add_argument("--scale", help="scale", type=float)
    parser.add_argument("--max_norm", help="max norm", type=float)
    args = parser.parse_args()

    config = yaml.load(open("config_debye.yaml", "r"), Loader=yaml.FullLoader)
    # Update the config with the provided arguments
    for key, value in vars(args).items():
        if value is not None:
            if "." in key:  # Handle nested keys
                keys = key.split(".")
                d = config
                for k in keys[:-1]:
                    d = d.setdefault(k, {})  # Traverse or create nested dictionaries
                d[keys[-1]] = type(d.get(keys[-1], value))(value)
            else:
                config[key] = type(config.get(key, value))(value)
    print(f"config: {config}")

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()