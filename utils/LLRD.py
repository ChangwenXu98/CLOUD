import torch
from torch import nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

"""Layer-wise learning rate decay"""

def roberta_base_AdamW_LLRD(model, lr, weight_decay):
    opt_parameters = []  # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters())
    print("number of named parameters =", len(named_parameters))

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # === Pooler and Regressor ======================================================

    params_0 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and any(nd in n for nd in no_decay)]
    print("params in pooler and regressor without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if ("pooler" in n or "Regressor" in n)
                and not any(nd in n for nd in no_decay)]
    print("params in pooler and regressor with decay =", len(params_1))

    head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(head_params)

    print("pooler and regressor lr =", lr)

    # === Hidden layers ==========================================================

    for layer in range(5, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} without decay =", len(params_0))
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} with decay =", len(params_1))

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)

        print("hidden layer", layer, "lr =", lr)

        lr *= 0.9

        # === Embeddings layer ==========================================================

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    print("params in embeddings layer without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    print("params in embeddings layer with decay =", len(params_1))

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(embed_params)
    print("embedding layer lr =", lr)

    return AdamW(opt_parameters, lr=lr)