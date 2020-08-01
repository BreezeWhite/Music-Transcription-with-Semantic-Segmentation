#!/bin/bash

# Example usage
MODEL_NAME=Maestro-Note
PRED_FOLDER=maestro_note
TH=7
MODE=note

MODEL_NAME=Maps-CFP-Frame
PRED_FOLDER=maps_cfp_frame
TH=0.01
MODE=frame

PRED_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_predictions.hdf"
LABEL_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_labels.pickle"

cd ..
python3 Evaluation.py $MODE \
    -p $PRED_PATH  \
    -l $LABEL_PATH \
    --onset-th $TH

echo "Onset threshold: ${TH}"
echo $PRED_FOLDER
