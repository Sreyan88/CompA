# COMPA: ADDRESSING THE GAP IN COMPOSITIONAL REASONING IN AUDIO-LANGUAGE MODELS

1. For Vanilla Training: 

Use the following command after updating the train and val file in "/src/laion_clap/train.sh" from the "src-stage1/laion_clap/" directory

```shell
sh train.sh
```

2. For training with compositionally-aware hard negatives:

Use the following command after updating the resume ckpt (the ckpt from vanilla training), train and val file in "/src-stage2/laion_clap/resume.sh" from the "src-stage2/laion_clap/" directory

```shell
sh resume.sh
```

3. For training with modular contrastive learning:

Use the following command after updating the resume ckpt (the ckpt from training with compositionally-aware hard negatives), train and val file in "/src-stage3/laion_clap/resume.sh" from the "src-stage3/laion_clap/" directory

```shell
sh resume.sh
```
