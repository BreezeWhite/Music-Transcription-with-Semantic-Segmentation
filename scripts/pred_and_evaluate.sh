#!/bin/bash

MODE=note
FEATURE_PATH=/data/Maps/test_feature
MODEL_PATH=./model/ICASSP-2019-Maps-Note
PRED_SAVE_PATH=./prediction/maps_note


cd ..
python3 Evaluation.py $MODE \
    -f $FEATURE_PATH \
    -m $MODEL_PATH \
    -s $PRED_SAVE_PATH
    
