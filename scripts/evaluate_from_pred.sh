#!/bin/bash

MODE=note
PRED_PATH=./maps_attn/Maps-Attn-W4.2.1_predictions.hdf
LABEL_PATH=./maps_attn/Maps-Attn-W4.2.1_labels.pickle

#PRED_PATH=/media/data/maps/test_feature/prediction/Maestro-Attn-W4.2_predictions.hdf
#LABEL_PATH=/media/data/maps/test_feature/prediction/Maestro-Attn-W4.2_labels.pickle

cd ..
#python3 Evaluation.py --help
python3 Evaluation.py $MODE \
    -p $PRED_PATH \
    -l $LABEL_PATH
    
