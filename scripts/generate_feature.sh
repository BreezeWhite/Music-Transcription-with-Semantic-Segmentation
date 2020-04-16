#!/bin/bash

DATASET=Maestro
D_PATH=/data/maestro-v1.0.0
PHASE=train
PIECE_PER_FILE=40
OUTPUT=/data/maestro-v1.0.0/${PHASE}_feature

cd ..
python3 GenFeature.py $DATASET $D_PATH\
    --phase $PHASE \
    --piece-per-file $PIECE_PER_FILE \
    --save-path $OUTPUT

