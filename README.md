### CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models

This is the official repository for the paper **CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models** accepted at ICLR 2024.

[[`Paper`](https://openreview.net/pdf?id=86NGO8qeWs)] [[`Checkpoint`]()] [[`Website`](https://sreyan88.github.io/compa_iclr/)]

## Setup
You are required to install the dependencies: `pip install -r requirements.txt`. If you have [conda](https://www.anaconda.com) installed, you can run the following: 

```shell
cd CompA && \
conda create -n compa python=3.10 && \
conda activate compa && \
pip install -r requirements.txt
```

## Training

1. For **Vanilla Training**: 

Use the following command after updating the train and val file in "/src/laion_clap/train.sh" from the "src-stage1/laion_clap/" directory

```shell
sh train.sh
```

2. For training with **compositionally-aware hard negatives**:

Use the following command after updating the resume ckpt (the ckpt from vanilla training), train and val file in "/src-stage2/laion_clap/resume.sh" from the "src-stage2/laion_clap/" directory

```shell
sh resume.sh
```

3. For training with **modular contrastive learning**:

Use the following command after updating the resume ckpt (the ckpt from training with compositionally-aware hard negatives), train and val file in "/src-stage3/laion_clap/resume.sh" from the "src-stage3/laion_clap/" directory

```shell
sh resume.sh
```

## Citation
```BibTex
@inproceedings{
ghosh2024compa,
title={CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models},
author={Sreyan Ghosh and Ashish Seth and Sonal Kumar and Utkarsh Tyagi and Chandra Kiran Reddy Evuru and Ramaneswaran S and S Sakshi and Oriol Nieto and Ramani Duraiswami and Dinesh Manocha},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=86NGO8qeWs}
}
```
