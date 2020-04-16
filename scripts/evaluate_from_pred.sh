#!/bin/bash

# Example usage
MODEL_NAME=ICASSP-2019-Maestro-Note
PRED_FOLDER=icassp_2019_maestro_note
TH=7
MODE=note

PRED_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_predictions.hdf"
LABEL_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_labels.pickle"

cd ..
python3 Evaluation.py $MODE \
    -p $PRED_PATH  \
    -l $LABEL_PATH \
    --onset-th $TH
    
echo $PRED_FOLDER
