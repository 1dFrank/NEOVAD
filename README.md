# Rethinking Open Vocabulary Video Anomaly Detection - Normality Matters
This repository contains the PyTorch implementation of our paper:  [Rethinking Open Vocabulary Video Anomaly Detection - Normality Matters]

![framework](./pic/framework.png)

---
## Setup
### Dependencies
Please set up the environment by following the `requirement.yml` file.

## Reproduce 

- Official Dataset Download
The original datasets for [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/), [ShanghaiTech](https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection), [XD-Violence](https://roc-ng.github.io/XD-Violence/), and [UBnormal](https://github.com/lilygeorgescu/UBnormal?tab=readme-ov-file) can be obtained from their official sources.

- Extract the CLIP feature
    The extracted CLIP features for datasets can be obtained from [CLIP features](https://drive.google.com/drive/folders/10i7ZvL6gAB2FwEdD8_ILPgtiOpu5CZi9?usp=drive_link).

    You can also use the CLIP model to extract features by referring to the scripts under `./scripts/feature_extract`.

### Inference
To reproduce the inference results:
- After downloading the dataset features, change the parameter `cfg.feat_prefix`  in `src/configs_base2novel.py` to your dataset feature path.

- Change the test list path in `src/configs_base2novel.py`, to fully/base/novel test set. The 'All' option is set by default in configs_base2novel.py.


- [Download ckpt](https://drive.google.com/drive/folders/1xWK8V0OW58BtBSNQwUl338OLm6OY47kJ?usp=drive_link) and move the folder to your own `ckpt/` , then run following command

     ```
    cd src
    python main.py --mode infer --dataset ucf --test best_ckpt --device cuda:0
    ```
The `--dataset` option can be `ucf`, `sh`, `xd`, or `ub`, referring to UCF-Crime, ShanghaiTech, XD-Violence, or UBnormal. 

### Training
If you want to training in scratch, The following files need to be modified in order to run the code on your own machine:

- Change the settings as mentioned above, keep the test list path to fully test set.

- for ucf-crime run training command:
```
python main.py --mode train --dataset ucf  --test best_ckpt seed 2 --device cuda:0
```
- for shanghaiTech run training command:
```
python main.py --mode train --dataset sh  --test best_ckpt seed 1 --device cuda:0
```
- for xd-violence run training command:
```
python main.py --mode train --dataset xd  --test best_ckpt seed 20 --device cuda:0
```
- for ubnormal run training command:
```
python main.py --mode train --dataset ub  --test best_ckpt seed 1 --device cuda:0
```

### Ablation study
For ablation study inference, Change the test list path in `src/configs_base2novel.py`, to fully/base/novel test set, run the following command:
- baseline (line 1)
```
python main.py --mode infer --dataset ucf --test baseline --device cuda:0
```
- w_GAT (line 2)
```
python main.py --mode infer --dataset ucf --test w_GAT --device cuda:0
```
- w_GAT_adapter (line 3)
```
python main.py --mode infer --dataset ucf --test w_GAT_adapter --device cuda:0
```
- wo_Lna (line 4)
```
python main.py --mode infer --dataset ucf --test wo_Lna --device cuda:0
```
- wo_Lod (line 5)
```
python main.py --mode infer --dataset ucf --test wo_Lod --device cuda:0
```
- full (line 6)
```
python main.py --mode infer --dataset ucf --test full --device cuda:0
```
For ablation study training, Change the test list path in `src/configs_base2novel.py`, to fully test set, run the following command:
- baseline (line 1)
```
python main.py --mode train --dataset ucf --test baseline --seed 2 --adapter False --temporal False --lamda2 0 --lamda3 0 --device cuda:0
```
- w_GAT (line 2)
```
python main.py --mode train --dataset ucf --test w_GAT --seed 2 --adapter False --temporal True --lamda2 0 --lamda3 0 --device cuda:0
```
- w_GAT_adapter (line 3)
```
python main.py --mode train --dataset ucf --test w_GAT_adapter --seed 2 --adapter True --temporal True --lamda2 0 --lamda3 0 --device cuda:0
```
- wo_Lna (line 4)
```
python main.py --mode train --dataset ucf --test wo_Lna --seed 2 --adapter True --temporal True --lamda2 1 --lamda3 0 --device cuda:0
```
- wo_Lod (line 5)
```
python main.py --mode train --dataset ucf --test wo_Lod --seed 2 --adapter True --temporal True --lamda2 0 --lamda3 2 --device cuda:0
```
- full (line 6)
```
python main.py --mode train --dataset ucf --test full --seed 2 --adapter True --temporal True --lamda2 1 --lamda3 2 --device cuda:0
```






