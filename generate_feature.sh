#!/bin/bash

DATASET=Maps
PHASE=test
PIECE_PER_FILE=20
OUTPUT=/media/whitebreeze/data/maps/test_feature

python3 GenFeature.py --help

python3 GenFeature.py $DATASET \
    --phase $PHASE \
    --piece-per-file $PIECE_PER_FILE \
    --save-path $OUTPUT
