#!/bin/bash

cd src

python main.py --config config_sacnn.yaml UNSUPVIDSEG.DATASET MOVi_A SOLVER.IMS_PER_BATCH 8 LOG_ID MOVi-A SOLVER.BASE_LR 0.00015 FLAGS.USE_CCPP False
