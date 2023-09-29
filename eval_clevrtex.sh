#!/bin/bash

cd src

python main.py --config config_sacnn.yaml --eval_only UNSUPVIDSEG.DATASET CM.M MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 6 MODEL.WEIGHTS /data/MOS_MODELS/TEX/SA-TEX-K6-LR/4567596/checkpoints/checkpoint_best_ARI_F.pth FLAGS.USE_CCPP False
