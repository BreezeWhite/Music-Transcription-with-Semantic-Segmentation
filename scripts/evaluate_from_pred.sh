#!/bin/bash

MODEL_NAME=feature-compare_harmonic
PRED_FOLDER=val_feature-compare_harmonic
TH=0.3

MODE=mpe_note
PRED_PATH="./prediction-paper/${PRED_FOLDER}/${MODEL_NAME}_predictions.hdf"
LABEL_PATH="./prediction-paper/${PRED_FOLDER}/${MODEL_NAME}_labels.pickle"

#PRED_PATH="./prediction/${PRED_FOLDER}/pred.hdf"
#LABEL_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_labels.pickle"


cd ..
python3 Evaluation.py $MODE \
    -p $PRED_PATH  \
    -l $LABEL_PATH \
    --onset-th $TH
    
echo $PRED_FOLDER
