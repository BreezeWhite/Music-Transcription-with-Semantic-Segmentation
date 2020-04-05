#!/bin/bash

DATASET=URMP
D_PATH=/data/URMP
PHASE=test
PIECE_PER_FILE=44
OUTPUT=~/${DATASET}/${PHASE}_feature

cd ..
#python3 GenFeature.py --help
python3 GenFeature.py $DATASET $D_PATH\
    --phase $PHASE \
    --piece-per-file $PIECE_PER_FILE \
    --save-path $OUTPUT
chown -R derek-wu ~/${DATASET}
mv ~/${DATASET} /host/home/AMT_Project
