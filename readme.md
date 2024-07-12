# Atari Pre-training Benchmark (Atari-PB)

Official repository for "Investigating Pre-Training Objectives for Generalization in Vision-Based Reinforcement Learning" (ICML 2024).

Atari-PB is the first benchmark to compare the generalization capabilities of pre-trained RL agents under a unified protocol.

Each algorithm is evaluated by first pre-training an agent on a 10M dataset (across 50 games) then fine-tuning on 3 distinct environment distributions (ID, Near-OOD, Far-OOD, total 65 games).

[[Paper]](https://arxiv.org/abs/2406.06037) [[Project page]](https://i-am-proto.github.io/atari-pb/) [[Model Weights & Datasets]](https://gofile.me/6WpIS/28juzliXx) (Password: ataripb)

## Installation

We assume that you have access to GPU (preferably multiple) that can run CUDA 11.8 and CuDNN 8.7.0.

### 1. Conda Environment

```
conda create -n ataripb python=3.9.11
conda activate ataripb
python3 -m pip install -r requirements.txt
AutoROM --accept-license
```

### 2. Dataset

While the above is running, you can start downloading required datasets for your experiments.

You don't have to download everything, especially the pre-training dataset (which ends up taking around 6~700GB of storage).

| Type      | Environment distribution            | Dataset source     | Download |
|-|-|-|-|
| Pre-train | In-Domain (ID)                      | [DQN-Replay-Dataset](https://offline-rl.github.io/) | `./scripts/download_pretrain_dataset.sh`   |
| Fine-tune | In-Domain (ID)                      | [DQN-Replay-Dataset](https://offline-rl.github.io/) | `./scripts/download_offline_bc_dataset.sh` |
| Fine-tune | Near-Out-of-Distribution (Near-OOD) | [DQN-Replay-Dataset](https://offline-rl.github.io/) | `./scripts/download_offline_bc_dataset.sh` |
| Fine-tune | Far-Out-of-Distribution (Far-OOD)   | 2M Rainbow agent                                    | [Download](https://gofile.me/6WpIS/fC465fvBE) (Password: ataripb) |

**Important**: You have to make several (cumbersome) changes to the scripts and configs for Atari-PB to know where the dataset is.

- Specify the download directory at `data_dir` when using download scripts.
- Specify the same directory at `replay_dataset_path` in the `./configs/dataloader/pretrain.yaml` configuration file.
- Specify your wandb entity name at `entity` in `./configs/pretrain.yaml`, `./configs/offline_bc.yaml`, etc. We recommend using `group_name` and `exp_name` as well.
- (Optional) Specify the directory to store the processed Atari-PB dataset at `ataripb_dataset_path` in `./configs/dataloader/pretrain.yaml`. By default, all datasets will be stored under `./ataripb/`.

## Implemented Algorithms

| Algorithm | Author / Paper | Pre-train script |
|-|-|-|
| CURL      | [Laskin et al.](https://arxiv.org/abs/2004.04136)          | `./scripts/pretrain/curl.sh`    |
| MAE       | [He et al.](https://arxiv.org/abs/2111.06377)              | `./scripts/pretrain/mae.sh`     |
| ATC       | [Stooke et al.](https://arxiv.org/abs/2009.08319)          | `./scripts/pretrain/atc.sh`     |
| SiamMAE   | [Gupta et al.](https://arxiv.org/abs/2305.14344)           | `./scripts/pretrain/siammae.sh` |
| R3M*      | [Nair et al.](https://arxiv.org/abs/2203.12601)            | `./scripts/pretrain/r3m.sh`     |
| BC        | [Pomerleau](https://www.ri.cmu.edu/pub_files/pub3/pomerleau_dean_1991_1/pomerleau_dean_1991_1.pdf) | `./scripts/pretrain/bc.sh` |
| SPR       | [Schwarzer et al.](https://arxiv.org/abs/2007.05929)       | `./scripts/pretrain/spr.sh`     |
| IDM       | [Christiano et al.](https://arxiv.org/abs/1610.03518)      | `./scripts/pretrain/idm.sh`     |
| SPR+IDM   | (SGI) [Schwarzer et al.](https://arxiv.org/abs/2106.04799) | `./scripts/pretrain/spr_idm.sh` |
| DT        | [Chen et al.](https://arxiv.org/abs/2106.01345)            | `./scripts/pretrain/dt.sh`      | 
| CQL       | [Kumar et al.](https://arxiv.org/abs/2006.04779)           | `./scripts/pretrain/cql_dist.sh` <br/> `./scripts/pretrain/cql_mse.sh`  |

Several algorithms likely won't fit into a single GPU; we recommend activating DDP via `--overrides num_gpus_per_node` in the scripts.

## Model Weights

You can download the checkpoints of our pre-trained models in the main experiment [here](https://gofile.me/6WpIS/zHpJWJLGu) (Password: ataripb).

For checkpoints in the ablation studies, please contact the author via quagmire@kaist.ac.kr.

To fine-tune these models, you can start with `./scripts/offline_bc/base.sh` and `./scripts/online_rl/base.sh`.