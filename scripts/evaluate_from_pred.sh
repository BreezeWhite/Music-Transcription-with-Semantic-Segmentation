#!/bin/bash

VERSION=1.0.2
MODEL_NAME=MusicNet-Attn-Note-Smooth-V$VERSION
PRED_FOLDER=musicnet_smooth_v$VERSION

MODE=note
PRED_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_predictions.hdf"
LABEL_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_labels.pickle"

#MODE=note
#PRED_PATH=./prediction/maps_attn/Maps-Attn-W4.2.1_predictions.hdf
#LABEL_PATH=./prediction/maps_attn/Maps-Attn-W4.2.1_labels.pickle

cd ..
#python3 Evaluation.py --help
python3 Evaluation.py $MODE \
    -p $PRED_PATH \
    -l $LABEL_PATH
    
