#!/bin/bash

MODE=onset
FEATURE_PATH=/media/whitebreeze/data/maps/test_feature2
MODEL_PATH=./model/Maestro-Attn-W4.2
PRED_SAVE_PATH=$FEATURE_PATH/prediction2

cd ..
python3 Evaluation.py --help
python3 Evaluation.py $MODE \
    -f $FEATURE_PATH \
    -m $MODEL_PATH \
    -s $PRED_SAVE_PATH
    
