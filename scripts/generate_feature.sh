#!/bin/bash

DATASET=Maps
PHASE=train
PIECE_PER_FILE=20
OUTPUT=/media/whitebreeze/data/maps/train_feature

cd ..
python3 GenFeature.py --help

python3 GenFeature.py $DATASET \
    --phase $PHASE \
    --piece-per-file $PIECE_PER_FILE \
    --save-path $OUTPUT
