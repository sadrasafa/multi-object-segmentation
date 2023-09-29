#!/bin/bash

cd src

python main.py --config config_sacnn.yaml --eval_only UNSUPVIDSEG.DATASET CM.R MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 8 MODEL.WEIGHTS /data/MOS_MODELS/CLEVR/SA-CLEVR-K8-LR/4566729/checkpoints/checkpoint_best_ARI_F.pth FLAGS.USE_CCPP False
