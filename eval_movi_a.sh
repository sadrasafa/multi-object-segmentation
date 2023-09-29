#!/bin/bash

cd src

python main.py --config config_sacnn.yaml --eval_only UNSUPVIDSEG.DATASET MOVi_A MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 6 MODEL.WEIGHTS /data/MOS_MODELS/MOVi-A/SA-MOVi_A-K6-LR/4566740/checkpoints/checkpoint_best_iou.pth FLAGS.USE_CCPP False
