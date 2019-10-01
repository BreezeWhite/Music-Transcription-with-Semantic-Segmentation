#!/bin/bash

DATASET=MusicNet
D_PATH=/media/data/MusicNet
PHASE=test
PIECE_PER_FILE=20
OUTPUT=/media/data/MusicNet/${PHASE}_feature

cd ..
#python3 GenFeature.py --help
python3 GenFeature.py $DATASET $D_PATH\
    --phase $PHASE \
    --piece-per-file $PIECE_PER_FILE \
    --save-path $OUTPUT
