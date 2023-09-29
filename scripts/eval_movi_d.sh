#!/bin/bash

cd ../src

python main.py --config config_sacnn.yaml --eval_only UNSUPVIDSEG.DATASET MOVi_D MODEL.MASK_FORMER.NUM_OBJECT_QUERIES 6 MODEL.WEIGHTS [PATH/TO/SEG/MODEL] FLAGS.USE_CCPP False
