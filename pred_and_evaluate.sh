#!/bin/bash

MODE=frame
FEATURE_PATH=/media/whitebreeze/data/maps/test_feature
MODEL_PATH=./attn_model
PRED_SAVE_PATH=$FEATURE_PATH/prediction

python3 Evaluation.py $MODE \
    -f $FEATURE_PATH \
    -m $MODEL_PATH \
    -s $PRED_SAVE_PATH
    
