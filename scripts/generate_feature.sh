#!/bin/bash

DATASET=Maestro
D_PATH=/data/Maestro
PHASE=train
PIECE_PER_FILE=400
OUTPUT=${D_PATH}/${PHASE}_feature/harmonic

cd ..
#python3 GenFeature.py --help
python3 GenFeature.py $DATASET $D_PATH\
    --phase $PHASE \
    --piece-per-file $PIECE_PER_FILE \
    --save-path $OUTPUT \
    --harmonic
chown -R derek-wu ${D_PATH}

