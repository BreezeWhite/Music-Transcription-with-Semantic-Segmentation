#!/bin/bash

MODE=note
PRED_PATH=./prediction/maps_attn/Maps-Attn-W4.2.1_predictions.hdf
LABEL_PATH=./prediction/maps_attn/Maps-Attn-W4.2.1_labels.pickle

cd ..
python3 Evaluation.py --help
python3 Evaluation.py $MODE \
    -p $PRED_PATH \
    -l $LABEL_PATH
    
