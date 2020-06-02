#!/bin/bash

DATASET=Rhythm
D_PATH=/data/sma_rhythm/76_pop_rhythm
PHASE=test
PIECE_PER_FILE=400
OUTPUT=${D_PATH}/${PHASE}_feature

cd ..
#python3 GenFeature.py --help
python3 GenFeature.py $DATASET $D_PATH\
    --phase $PHASE \
    --piece-per-file $PIECE_PER_FILE \
    --save-path $OUTPUT
chown -R derek-wu ${D_PATH}

