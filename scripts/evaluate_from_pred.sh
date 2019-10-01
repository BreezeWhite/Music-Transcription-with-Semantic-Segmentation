#!/bin/bash

MODE=note
PRED_PATH=./prediction/musicnet_attn/MusicNet-Attn-W4.2_predictions.hdf
LABEL_PATH=./prediction/musicnet_attn/MusicNet-Attn-W4.2_labels.pickle


cd ..
#python3 Evaluation.py --help
python3 Evaluation.py $MODE \
    -p $PRED_PATH \
    -l $LABEL_PATH
    
