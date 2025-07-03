## CLOUD ##

#### CLOUD: A Scalable and Physics-Informed Foundation Model for Crystal Representation Learning [[arXiv]](https://arxiv.org/abs/2506.17345)

#### Workshop paper in Foundation Models for Science: Progress, Opportunities, and Challenges at NeurIPS 2024 [[Paper]](https://openreview.net/forum?id=geZ5LQOCSj) </br>
[Changwen Xu](https://changwenxu98.github.io/), [Shang Zhu](https://shang-zhu.github.io/), [Venkatasubramanian Viswanathan](https://aero.engin.umich.edu/people/viswanathan-venkat/) </br>
University of Michigan </br>

This is the official implementation of <strong><em>CLOUD</em></strong>: ["CLOUD: A Scalable and Physics-Informed Foundation Model for Crystal Representation Learning"](https://arxiv.org/abs/2506.17345). In this work, we introduce Crystal Language mOdel for Unified and Differentiable materials modeling (CLOUD), a Transformer-based foundation model for crystal representation learning via a novel Symmetry-Consistent Ordered Parameter Encoding (SCOPE) and accurate, generalizable, and scalable property prediction. We further extend the model by integrating with Debye model for thermodynamic-consistent prediction of phonon-related properties. If you find our work useful in your research, please cite:
```
@article{xu2025cloud,
  title={A Scalable and Physics-Informed Foundation Model for Crystal Representation Learning},
  author={Xu, Changwen and Zhu, Shang and Viswanathan, Venkatasubramanian},
  journal={arXiv preprint arXiv:2506.17345},
  year={2025}
}

@inproceedings{xu2024cloud,
  title={CLOUD: A Scalable Scientific Foundation Model for Crystal Representation Learning},
  author={Xu, Changwen and Zhu, Shang and Viswanathan, Venkatasubramanian},
  booktitle={Neurips 2024 Workshop Foundation Models for Science: Progress, Opportunities, and Challenges}
}
```

## Getting Started

### Installation

Set up conda environment and clone the github repo

```
# clone the source code of CLOUD
$ git clone https://github.com/ChangwenXu98/CLOUD.git
$ cd CLOUD

# create the environment from environment.yml
$ conda env create -f environment.yml
$ conda activate cloud
```

## Run the Model

## Convert crystal structures to SCOPE representation
To obtain the string representation from cif files.
```
$ python structure_to_str.py --dir <path_to_cif> --out <output_path> --numproc <num_of_processes> --batchsize <batch_size>
```

### Pretraining
To pretrain CLOUD, where the configurations and detailed explaination for each variable can be found in `config_pretrain.yaml`.
```
$ python -m torch.distributed.launch --nproc_per_node=2 pretrain.py
```
<em>DistributedDataParallel</em> is used for faster pretraining.

The checkpoints of the pretrained model and the pretraining data can be found [here](https://drive.google.com/drive/folders/1-ve6g7f4BWVRkUjKMZP4Q-YmjF2hHOXx?usp=sharing).

### Finetuning
To finetune the pretrained CLOUD on MatBench or UnconvBench about crystal properties, where the configurations and detailed explaination for each variable can be found in `config.yaml`.
```
$ python train.py
```

To finetune the pretrained CLOUD on MatBench Discovery and make predictions for WBM test set.
```
$ python train_mp.py
$ python wbm_predict.py
```

### Integrating CLOUD with physics laws
To demonstrate the capability of integrating CLOUD with physics laws in a differentiable physics framework for physics-consistent property predictions, we develop CLOUD-DEBYE in which the Debye model is implemented with Gauss–Legendre quadrature to enable training from end to end. 

To finetune the pretrained CLOUD encoder with phonon internal energy (U) or constant-volume heat capacity (Cv) labels with CLOUD-DEBYE, where the configurations and detailed explaination for each variable can be found in `config_debye.yaml`.
```
$ python train_debye.py
```