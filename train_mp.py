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
from dataset.dataset_mp import CloudDatasetWrapper
from model.cloud import Cloud
import shutil
from utils.LLRD import roberta_base_AdamW_LLRD


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        if config["save_path"] is None:
            self.log_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S'))
        else:
            self.log_dir = os.path.join('runs', config["save_path"])
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.dataset = CloudDatasetWrapper(**config['dataset'])
        self.criterion = nn.L1Loss()

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        print("Running on:", device)

        return device
    
    @staticmethod
    def _save_config_file(ckpt_dir):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            shutil.copy('./config_mp.yaml', os.path.join(ckpt_dir, 'config_mp.yaml'))

    def train(self):
        loss_fold = []
        train_dataset, valid_dataset, train_loader, valid_loader = self.dataset.get_data_loaders()
        print(f"mu: {train_dataset.mu}")
        print(f"std: {train_dataset.std}")

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

        ckpt_dir = os.path.join(self.writer.log_dir, 'checkpoints')
        self._save_config_file(ckpt_dir)

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
                optimizer.step()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=n_iter)
                    print(epoch_counter, bn, 'loss', loss.item())
                n_iter += 1

            print("Start Validation")
            valid_loss = self._validate(model, valid_loader, train_dataset.mu, train_dataset.std)
            self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
            print('Validation', epoch_counter, 'MAE Loss', valid_loss)

            states = {
                'model_state_dict': model.state_dict(),
                "best_valid_loss": best_valid_loss,
                'epoch': epoch_counter,
                'mu': train_dataset.mu,
                'std': train_dataset.std,
            }

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(states, os.path.join(ckpt_dir, 'model_best.pth'))

        
            torch.save(states, os.path.join(ckpt_dir, 'model.pth'))

            valid_n_iter += 1

        loss_fold.append(best_valid_loss)
        print("best loss:", best_valid_loss)

    def _validate(self, model, valid_loader, mu, std):
        valid_loss = 0
        with torch.no_grad():
            model.eval()
            for bn, batch in enumerate(tqdm(valid_loader)):
                _, output, target = self.loss_fn(model, batch)
                output = output * std + mu
                target = target * std + mu
                loss = self.criterion(output.squeeze(), target.squeeze())
                valid_loss += loss.item()

            valid_loss /= len(valid_loader)
            print("Valid Loss: {:.4f}".format(valid_loss))
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
        output = model(input_ids, attention_mask)
        
        loss = self.criterion(output.squeeze(), target.squeeze())
        
        return loss, output, target
    
def main():
    config = yaml.load(open("config_mp.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()