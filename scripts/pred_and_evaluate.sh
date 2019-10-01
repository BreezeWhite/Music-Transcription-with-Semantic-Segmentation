#!/bin/bash

MODE=note
FEATURE_PATH=/data/MusicNet/test_feature
MODEL_PATH=./model/MusicNet-Attn-W4.2
PRED_SAVE_PATH=./prediction/musicnet_attn


cd ..
python3 Evaluation.py --help
python3 Evaluation.py $MODE \
    -f $FEATURE_PATH \
    -m $MODEL_PATH \
    -s $PRED_SAVE_PATH
    
