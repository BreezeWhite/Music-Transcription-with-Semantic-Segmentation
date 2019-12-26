#!/bin/bash

MODEL_NAME=Maestro-Attn-Note-Smooth
PRED_FOLDER=maestro_attn_note_smooth
TH=6

MODEL_NAME=Maestro-Smooth-Ultimate-Attn
PRED_FOLDER=maestro_smooth_ultimate_attn
TH=5.5

MODEL_NAME=MusicNet-Smooth-Ultimate-Attn
PRED_FOLDER=musicnet_smooth_ultimate_attn
TH=6

#MODEL_NAME=Maestro-Attn-V4.2.1
#PRED_FOLDER=maestro_attn_v4.2.1
#TH=5

MODEL_NAME=MusicNet-Attn-Note-Smooth-V1.0.2
PRED_FOLDER=musicnet_smooth_v1.0.2
TH=8

#MODEL_NAME=Maestro-Maps-V4.2.1
#PRED_FOLDER=maestro_maps_v4.2.1
#TH=7.5

#MODEL_NAME=ICASSP-2019-Maestro-Note
#PRED_FOLDER=icassp_2019_maestro_note
#TH=7

#MODEL_NAME=ICASSP-2019-Maestro-Frame
#PRED_FOLDER=icassp_2019_maestro_frame

MODE=note
PRED_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_predictions.hdf"
LABEL_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_labels.pickle"

#PRED_PATH="./prediction/${PRED_FOLDER}/pred.hdf"
#LABEL_PATH="./prediction/${PRED_FOLDER}/${MODEL_NAME}_labels.pickle"


cd ..
python3 Evaluation.py $MODE \
    -p $PRED_PATH  \
    -l $LABEL_PATH \
    --onset-th $TH
    
echo $PRED_FOLDER
