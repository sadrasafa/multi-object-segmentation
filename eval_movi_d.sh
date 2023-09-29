#!/bin/bash

cd src

python main.py --config config_sacnn.yaml --eval_only UNSUPVIDSEG.DATASET MOVi_D MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 6 MODEL.WEIGHTS /data/MOS_MODELS/MOVi-D/SA-MOVi_D-K6-LR/4556550/checkpoints/checkpoint_best_ARI_F.pth FLAGS.USE_CCPP False
