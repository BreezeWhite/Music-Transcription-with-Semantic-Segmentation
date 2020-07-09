#!/bin/bash

DATASET=Maps
D_PATH=/data/Maps
PHASE=test
PIECE_PER_FILE=60
OUTPUT=${D_PATH}/${PHASE}_feature/harmonic

cd ..
#python3 GenFeature.py --help
python3 GenFeature.py $DATASET $D_PATH\
    --phase $PHASE \
    --piece-per-file $PIECE_PER_FILE \
    --save-path $OUTPUT \
    --harmonic
chown -R derek-wu ${D_PATH}

