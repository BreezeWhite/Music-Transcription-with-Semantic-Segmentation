#!/bin/bash

MODE=frame
PRED_PATH=/media/whitebreeze/data/maps/test_feature/prediction/attn_model_predictions.hdf
LABEL_PATH=/media/whitebreeze/data/maps/test_feature/prediciton/attn_model_labels.hdf

python3 Evaluation.py $MODE \
    -p $PRED_PATH \
    -l $LABEL_PATH
    
