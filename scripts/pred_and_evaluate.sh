#!/bin/bash

MODE=frame
FEATURE_PATH=/data/Maps/test_feature
MODEL_PATH=./model/Maps-Attn-W4.2.1
PRED_SAVE_PATH=$FEATURE_PATH/prediction
PRED_SAVE_PATH=./prediction/maps_attn

cd ..
python3 Evaluation.py --help
python3 Evaluation.py $MODE \
    -f $FEATURE_PATH \
    -m $MODEL_PATH \
    -s $PRED_SAVE_PATH
    
