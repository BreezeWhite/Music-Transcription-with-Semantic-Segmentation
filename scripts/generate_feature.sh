#!/bin/bash

DATASET=MusicNet
D_PATH=/data/MusicNet
PHASE=test
PIECE_PER_FILE=10
OUTPUT=${D_PATH}/${PHASE}_feature/harmonic


cd ..
python3 GenFeature.py $DATASET $D_PATH\
    --phase $PHASE \
    --piece-per-file $PIECE_PER_FILE \
    --save-path $OUTPUT \
    --harmonic

