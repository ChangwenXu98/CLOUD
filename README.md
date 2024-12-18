## CLOUD ##

#### Foundation Models for Science: Progress, Opportunities, and Challenges at NeurIPS 2024 [[Paper]](https://openreview.net/forum?id=geZ5LQOCSj) </br>
[Changwen Xu](https://changwenxu98.github.io/), [Shang Zhu](https://shang-zhu.github.io/), [Venkatasubramanian Viswanathan](https://aero.engin.umich.edu/people/viswanathan-venkat/) </br>
University of Michigan </br>

This is the official implementation of <strong><em>CLOUD</em></strong>: ["CLOUD: A Scalable Scientific Foundation Model for Crystal Representation Learning"](https://openreview.net/forum?id=geZ5LQOCSj). In this work, we introduce CrystaL fOUnDation model (CLOUD), a Transformer-based foundation model for crystal representation learning via a novel symmetry-aware string representation and accurate, generalizable, and scalable property prediction. If you find our work useful in your research, please cite:
```
@inproceedings{xu2024cloud,
  title={CLOUD: A Scalable Scientific Foundation Model for Crystal Representation Learning},
  author={Xu, Changwen and Zhu, Shang and Viswanathan, Venkatasubramanian},
  booktitle={Neurips 2024 Workshop Foundation Models for Science: Progress, Opportunities, and Challenges}
}
```

This work is still under development and new progress will be made publically available when ready.

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

## Convert crystal structures to symmetry-aware string representation
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
