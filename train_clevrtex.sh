#!/bin/bash

cd src

python main.py --config config_sacnn.yaml UNSUPVIDSEG.DATASET CM.M SOLVER.IMS_PER_BATCH 8 LOG_ID CLEVRTEX SOLVER.BASE_LR 0.00015 FLAGS.USE_CCPP False
