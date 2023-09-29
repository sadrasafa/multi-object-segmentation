# Multi-Object Discovery by Low-Dimensional Object Motion (ICCV 2023)

### [Project Page](https://kuis-ai.github.io/monodepthseg/) | [Paper](https://arxiv.org/abs/2110.11275)


This is the code for our ICCV 2023 paper:
> **[Multi-Object Discovery by Low-Dimensional Object Motion](https://arxiv.org/abs/2110.11275)** \
> [Sadra Safadoust](https://sadrasafa.github.io/) and [Fatma Güney](https://mysite.ku.edu.tr/fguney/)


# Requirements

1. Create the conda environment and install the requirements:
```
conda create -n mos python=3.8
conda activate mos
conda install -y pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch
conda install -y kornia jupyter tensorboard timm einops scikit-learn scikit-image openexr-python tqdm -c conda-forge
conda install -y gcc_linux-64=7 gxx_linux-64=7 fontconfig matplotlib
pip install cvbase opencv-python filelock
```

2. Install Mask2Former:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd src/mask2former/modeling/pixel_decoder/ops
sh make.sh
```
If you face any problems while installing Mask2Former, please refer to the installation steps in the [Mask2Former](https://github.com/facebookresearch/Mask2Former) and [Detectron2](https://github.com/facebookresearch/detectron2) repositories.

# Datasets

## CLEVR/ClevrTex and MOVi 

Please follow the steps [here](https://github.com/karazijal/probable-motion#data-preparation) for the CLEVR, ClevrTex, and MOVi datasets.

## DAVIS2017

Download and extract the [DAVIS2017 dataset](https://davischallenge.org/davis2017/code.html) into `src/data/DAVIS2017`. Download the motion annotations from [here](https://github.com/Jyxarthur/OCLR_model) and extract them into `/src/data/DAVIS2017/Annotations_unsupervised_motion`. Use [motion-grouping](https://github.com/charigyang/motiongrouping) repository to create the flows for this dataset.

## KITTI

Download the [KITTI raw dataset](https://www.cvlibs.net/datasets/kitti/user_login.php) into `src/data/KITTI/KITTI-Raw`. Calculate the flows using [RAFT](https://github.com/princeton-vl/RAFT) into `src/data/KITTI/RAFT_FLOWS`. Note that we use `png` flows for KITTI. Download segmentation labels from [this repository](https://github.com/zpbao/Discovery_Obj_Move).

---

The data directory structure should look like this:
```
├── src
    ├── data
        ├── movi_a
            ├── train
            ├── validation
        ├── movi_c
            ├── train
            ├── validation
        ├── movi_d
            ├── train
            ├── validation
        ├── movi_d
            ├── train
            ├── validation
        ├── moving_clevr
            ├── tar
                ├── CLEVRMOV_clevr_v2_*.tar
                ...
        ├── moving_clevrtex
            ├── tar
                ├── CLEVRMOV_full_old_ten_slow_short_*.tar
                ...
        ├── DAVIS2017
            ├── JPEGImages
            ├── Annotations_unsupervised_motion
            ├── Flows_gap4
            ...
        ├── KITTI
            ├── KITTI-Raw
                ├── 2011_09_26
                ...
            ├── RAFT_FLOWS
                ├── 2011_09_26
                ...
            ├── KITTI_test

```


# Training

You can train the model on the datasets by running the corresponding scripts.
E.g. for movi_c, run `./scripts/train_movi_c.sh`

# Inference

You can evaluate the model on the synthetic datasets by running the corresponding scripts.
E.g. for movi_c, run `./scripts/eval_movi_c.sh` and set the model weights path in the script to the path of the segmentation model.

# Trained Models

You can download the trained models from [here](https://drive.google.com/drive/folders/1d2LcexNE_bA5bmrss6t6f1UauVzLIqVc?usp=sharing).

# Acknowledgements
The structure of this code is largely based on [probable-motion](https://github.com/karazijal/probable-motion)  repository. Many thanks to the authors for sharing their code.