#!/bin/bash

MODEL_NAME=Dilated-Conv-MusicNet-Note-Smooth
PRED_FOLDER=dilated_conv_musicnet_note_smooth

MODEL_NAME=Maestro-Attn-V4.2.1
PRED_FOLDER=maestro_attn_v4.2.1

MODEL_NAME=Maps-Attn-W4.2.1
PRED_FOLDER=maps_attn

MODE=note
PRED_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_predictions.hdf"
LABEL_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_labels.pickle"

cd ..
python3 Evaluation.py $MODE \
    -p $PRED_PATH \
    -l $LABEL_PATH
    
