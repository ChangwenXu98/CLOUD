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
    

    def forward(self, input_ids, attention_mask, T=300, n_atoms=2, n=20, mu=1500.0, scale=1200.0, task="cv", return_theta=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        theta = self.Regressor(logits).squeeze()
        # theta = torch.exp((theta * scale + mu) * np.log(10))
        # theta = torch.pow(10, theta)
        theta = theta * scale + mu
        if return_theta:
            return theta
        # theta_log10 = self.Regressor(logits).squeeze()
        # # theta = torch.pow(10, theta_log10)  # Ensure differentiability
        # theta = torch.exp(theta_log10 * np.log(10))
        # print(f"Theta min: {theta.min().item()}, Theta max: {theta.max().item()}")
        # theta = torch.clamp(theta, min=1, max=10000)  # Prevents extreme values
        # print(f"After clamp, Theta min: {theta.min().item()}, Theta max: {theta.max().item()}")
        if task == "cv":
            return self.Cv(theta=theta, T=T, n_atoms=n_atoms, n=n)
        elif task == "internal_energy":
            return self.U(theta=theta, T=T, n_atoms=n_atoms, n=n)
    
    def Cv(self, theta, T, n_atoms, n=20):
        """Compute heat capacity with PyTorch to allow backpropagation."""
        if T == 0:
            return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)  # Avoid division by zero

        x_D = theta / T  # Upper limit of integration
        
        # Get Gauss-Legendre quadrature points and weights using PyTorch tensors
        x_np, w_np = np.polynomial.legendre.leggauss(n)
        
        x = torch.tensor(x_np, dtype=theta.dtype, device=theta.device).unsqueeze(0)  # Shape: [1, n]
        w = torch.tensor(w_np, dtype=theta.dtype, device=theta.device).unsqueeze(0)  # Shape: [1, n]
        
        x_D = x_D.unsqueeze(-1)  # Shape: [batch, 1]

        # Transform x from [-1,1] to [0,x_D]
        t = 0.5 * x_D * (x + 1)  # Rescale quadrature points
        dt_dx = 0.5 * x_D  # Jacobian of the transformation
        
        # Compute the function values at quadrature points using PyTorch
        f = (t**4 * torch.exp(t)) / (torch.exp(t) - 1)**2
        
        # Perform the quadrature integration
        integral = torch.sum(w * f, dim=-1) * dt_dx.squeeze(-1)
        
        # Compute C_v using PyTorch tensors
        C_v = 9 * (T / theta) ** 3 * integral * 8.314 * n_atoms  

        return C_v
    
    def U(self, theta, T, n_atoms, n=20):
        """Compute heat capacity with PyTorch to allow backpropagation."""
        if T == 0:
            return torch.tensor(0.0, device=theta.device, dtype=theta.dtype)  # Avoid division by zero

        x_D = theta / T  # Upper limit of integration
        
        # Get Gauss-Legendre quadrature points and weights using PyTorch tensors
        x_np, w_np = np.polynomial.legendre.leggauss(n)
        
        x = torch.tensor(x_np, dtype=theta.dtype, device=theta.device).unsqueeze(0)  # Shape: [1, n]
        w = torch.tensor(w_np, dtype=theta.dtype, device=theta.device).unsqueeze(0)  # Shape: [1, n]
        
        x_D = x_D.unsqueeze(-1)  # Shape: [batch, 1]

        # Transform x from [-1,1] to [0,x_D]
        t = 0.5 * x_D * (x + 1)  # Rescale quadrature points
        dt_dx = 0.5 * x_D  # Jacobian of the transformation
        
        # Compute the function values at quadrature points using PyTorch
        f = (t**3) / (torch.exp(t) - 1)
        
        # Perform the quadrature integration
        integral = torch.sum(w * f, dim=-1) * dt_dx.squeeze(-1)
        
        # zero-point energy
        U_0 = 9/8 * 8.314 * n_atoms * theta
        
        # Compute C_v using PyTorch tensors
        U = 9 * (T / theta) ** 3 * integral * 8.314 * n_atoms * T + U_0

        return U