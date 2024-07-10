### CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models

This is the official repository for the paper **CompA: Addressing the Gap in Compositional Reasoning in Audio-Language Models** accepted at ICLR 2024.

[[`Website`](https://sreyan88.github.io/compa_iclr/)] [[`Paper`](https://openreview.net/pdf?id=86NGO8qeWs)] [[`CompA-order`](https://drive.google.com/file/d/1A_HDH0sO6Pp-kvdcTJrAA6MJZiItHZTQ/view?usp=sharing)] [[`CompA-attribute`](https://drive.google.com/file/d/1vWpq2fTcT8T7ec8pZ_EG2v29PwJPfcJm/view?usp=sharing)] [[`CompA-661k`](https://drive.google.com/file/d/1FIC9XBQStw9EUBVxKJXVKTQpPIkfy0cZ/view?usp=sharing)] [[`Stage2 CSV`](https://drive.google.com/file/d/1yDqkEYZi44yqXJPLw50bacmhkgRJJ_0b/view?usp=sharing)] [[`Stage3 CSV`](https://drive.google.com/file/d/1GF2Fc-sQUGA52LXwfLjNSL4Hl481BqYm/view?usp=sharing)]

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

Use the following command after updating the train and val file in `/src/laion_clap/train.sh` from the `src-stage1/laion_clap/` directory:

```shell
sh train.sh
```

2. For training with **compositionally-aware hard negatives**:

Use the following command after updating the resume ckpt (the ckpt from vanilla training), train and val file in `/src-stage2/laion_clap/resume.sh` from the `src-stage2/laion_clap/` directory

```shell
sh resume.sh
```

3. For training with **modular contrastive learning**:

To generate the training file for stage 3 training, run the following command. The demo files required to run the code have been shared as well.

```shell
cd audio_mix
python pos-neg-generation.py -audio ./data/final_audio.csv -sound_class ./data/global_sound.yml
```

Use the following command after updating the resume ckpt (the ckpt from training with compositionally-aware hard negatives), train and val file in `/src-stage3/laion_clap/resume.sh` from the `src-stage3/laion_clap/` directory:

```shell
sh resume.sh
```

### Evaluation

The evaluation files need `hook.py` from [CLAP](https://github.com/LAION-AI/CLAP) repository. Please place the files in the `CLAP/src/laion_clap/` folder and run the below commands as required.

1. For **Zero-Shot evaluation**:

```shell
python ./evaluation/zshot.py <test_files_dir_path> <class_label_to_idx_file_path> <clap_ckpt_path>
```
`test_files_dir_path` - Path to the folder which contains audio files and their respective jsons. This format can be obtained by using the [audio-dataset](https://github.com/LAION-AI/audio-dataset/tree/main) repo.
<br>
`class_label_to_idx_file_path` - Path to the file which contains a class label and its respective index in the format of a Python dictionary. These files can be found in `CLAP/class_labels`.

2. For **CompA-Order and CompA-Attribute evaluations**:

```shell
python ./evaluation/benchmark_eval.py <benchmark_file_path> <audio_dir_path> <clap_ckpt_path>
```

benchmark_file_path - Path to CompA-Order or CompA-Attribute benchmark file.<br>
audio_dir_path - After downloading and extracting from the link provided, the path to CompA_order_files or CompA_attribute_files folder.

## 🌻 Acknowledgement
This repository benefits from [CLAP](https://github.com/LAION-AI/CLAP). Thanks for their awesome work.


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
