#!/bin/bash

cd src

python main.py --config config_sacnn.yaml --eval_only UNSUPVIDSEG.DATASET MOVi_C MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 6 MODEL.WEIGHTS /data/MOS_MODELS/MOVi-C/SA-MOVi_C-K6-LR/4556549/checkpoints/checkpoint_best_ARI_F.pth FLAGS.USE_CCPP False
